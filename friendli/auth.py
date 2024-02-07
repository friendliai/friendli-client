# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Auth Tools."""

from __future__ import annotations

import functools
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from requests import HTTPError, JSONDecodeError
from typing_extensions import TypeAlias

import friendli
from friendli.di.injector import get_injector
from friendli.errors import APIError, AuthorizationError, AuthTokenNotFoundError
from friendli.logging import logger
from friendli.utils.fs import get_friendli_directory
from friendli.utils.request import DEFAULT_REQ_TIMEOUT, decode_http_err
from friendli.utils.url import URLProvider

access_token_path = get_friendli_directory() / "access_token"
refresh_token_path = get_friendli_directory() / "refresh_token"
mfa_token_path = get_friendli_directory() / "mfa_token"


class TokenType(str, Enum):
    """Token type enums."""

    ACCESS = "ACCESS"
    REFRESH = "REFRESH"
    MFA = "MFA"


token_path_map = {
    TokenType.ACCESS: access_token_path,
    TokenType.REFRESH: refresh_token_path,
    TokenType.MFA: mfa_token_path,
}

ResponseBody: TypeAlias = Union[Dict[str, Any], List[Dict[str, Any]], None]


def get_auth_header(
    token: Optional[str] = None, team_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get authorization header.

    Returns:
        Dict[str, Any]: HTTP Authorization headers for the request.

    """
    token_: Optional[str]

    token_from_cfg = get_token(TokenType.ACCESS)
    token_from_env = friendli.token

    if token is not None:
        token_ = token
    elif token_from_env:
        if token_from_cfg:
            logger.warning(
                "You've entered your login information in two places - through the "
                "'FRIENDLI_TOKEN' environment variable and the 'friendli login' CLI "
                "command. We will use the access token from the 'FRIENDLI_TOKEN' "
                "environment variable and ignore the login session details. This might "
                "lead to unexpected authorization errors. If you prefer to use the "
                "login session instead, unset the 'FRIENDLI_TOKEN' environment "
                "variable. If you don't want to see this warning again, run "
                "'friendli logout' to remove the login session."
            )
        token_ = token_from_env
    else:
        token_ = token_from_cfg

    if token_ is None:
        raise AuthTokenNotFoundError(
            "Should set 'FRIENDLI_TOKEN' environment variable or sign in with 'friendli login'."
        )

    headers = {"Authorization": f"Bearer {token_}"}

    if team_id is not None:
        headers["X-Friendli-Team"] = team_id
    elif friendli.team_id:
        headers["X-Friendli-Team"] = friendli.team_id

    return headers


def get_token(token_type: TokenType) -> Union[str, None]:
    """Get an auth token."""
    try:
        if token_type == TokenType.ACCESS:
            return access_token_path.read_text()
        if token_type == TokenType.REFRESH:
            return refresh_token_path.read_text()
        if token_type == TokenType.MFA:
            return mfa_token_path.read_text()
        raise ValueError("token_type should be one of 'access' or 'refresh' or 'mfa'.")
    except FileNotFoundError:
        return None


def update_token(token_type: TokenType, token: str) -> None:
    """Update saved token value."""
    token_path_map[token_type].write_text(token)


def delete_token(token_type: TokenType) -> None:
    """Delete token."""
    token_path_map[token_type].unlink(missing_ok=True)


def clear_tokens() -> None:
    """Clear all tokens."""
    for token_type in TokenType:
        delete_token(token_type)


# pylint: disable=too-many-nested-blocks
def safe_request(func: Callable[..., requests.Response]) -> Callable[..., Any]:
    """Decorator for automatic token refresh."""

    @functools.wraps(func)
    def inner(*args, **kwargs) -> Any:
        injector = get_injector()
        url_provider = injector.get(URLProvider)

        resp = func(*args, **kwargs)
        if resp.status_code in (401, 403):
            refresh_token = get_token(TokenType.REFRESH)
            if refresh_token is not None:
                refresh_r = requests.post(
                    url_provider.get_web_backend_uri("/api/auth/cli/refresh_token"),
                    json={"refresh_token": refresh_token},
                    timeout=DEFAULT_REQ_TIMEOUT,
                )
                try:
                    refresh_r.raise_for_status()
                except requests.HTTPError as exc:
                    raise AuthorizationError(
                        "Failed to refresh access token. Please login again."
                    ) from exc

                update_token(
                    token_type=TokenType.ACCESS, token=refresh_r.json()["accessToken"]
                )
                update_token(
                    token_type=TokenType.REFRESH,
                    token=refresh_r.json()["refreshToken"],
                )
                # We need to restore file offset if we want to transfer file objects
                if "files" in kwargs:
                    files = kwargs["files"]
                    for _, file_tuple in files.items():
                        for element in file_tuple:
                            if hasattr(element, "seek"):
                                # Restore file offset
                                element.seek(0)
                resp = func(*args, **kwargs)
                resp.raise_for_status()
            else:
                raise AuthorizationError(
                    "Failed to refresh access token. Please login again."
                )
        else:
            try:
                resp.raise_for_status()
            except HTTPError as exc:
                raise APIError(decode_http_err(exc)) from exc
        try:
            return resp.json()
        except JSONDecodeError:
            return None

    return inner
