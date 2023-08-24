# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Auth Tools."""

from __future__ import annotations

import functools
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from requests import HTTPError, JSONDecodeError
from typing_extensions import TypeAlias

import periflow
from periflow.di.injector import get_injector
from periflow.errors import APIError, AuthorizationError, AuthTokenNotFoundError
from periflow.utils.fs import get_periflow_directory
from periflow.utils.request import DEFAULT_REQ_TIMEOUT, decode_http_err
from periflow.utils.url import URLProvider

access_token_path = get_periflow_directory() / "access_token"
refresh_token_path = get_periflow_directory() / "refresh_token"
mfa_token_path = get_periflow_directory() / "mfa_token"


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


def get_auth_header() -> Dict[str, Any]:
    """Get authorization header.

    Returns:
        Dict[str, Any]: HTTP Authorization headers for the request.

    """
    token: Optional[str]
    if periflow.api_key:
        token = periflow.api_key
    else:
        token = get_token(TokenType.ACCESS)

    if token is None:
        raise AuthTokenNotFoundError(
            "Should set PERIFLOW_API_KEY environment variable or sign in with 'pf login'."
        )

    return {"Authorization": f"Bearer {token}"}


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
                    url_provider.get_training_uri("token/refresh/"),
                    data={"refresh_token": refresh_token},
                    timeout=DEFAULT_REQ_TIMEOUT,
                )
                try:
                    refresh_r.raise_for_status()
                except requests.HTTPError as exc:
                    raise AuthorizationError(
                        "Failed to refresh access token. Please login again."
                    ) from exc

                update_token(
                    token_type=TokenType.ACCESS, token=refresh_r.json()["access_token"]
                )
                update_token(
                    token_type=TokenType.REFRESH,
                    token=refresh_r.json()["refresh_token"],
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
