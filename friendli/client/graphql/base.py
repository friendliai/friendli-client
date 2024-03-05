# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli GQL Client Service."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx
from gql import Client, gql
from gql.transport.exceptions import TransportError, TransportQueryError
from gql.transport.httpx import HTTPXTransport

from friendli.auth import TokenType, get_auth_header, get_token, update_token
from friendli.di.injector import get_injector
from friendli.errors import APIError, AuthenticationError, AuthorizationError
from friendli.utils.url import URLProvider


def get_default_gql_client(url: Optional[str] = None) -> Client:
    """Get a default GraphQL client."""
    if url is None:
        injector = get_injector()
        url_provider = injector.get(URLProvider)
        url = url_provider.get_web_backend_uri("api/graphql")

    gql_transport = HTTPXTransport(url=url, headers=get_auth_header())
    gql_client = Client(transport=gql_transport, fetch_schema_from_transport=True)
    return gql_client


class GqlClient:
    """Base interface of graphql client to Friendli system."""

    def __init__(
        self,
        url: Optional[str] = None,
        max_retries: int = 3,
        retry_interval: float = 1,
    ) -> None:
        """Initialize GqlClient."""
        self._client = get_default_gql_client(url)
        self._max_retries = max_retries
        self._retry_interval = retry_interval

    def run(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run graphql."""
        attempts = 0
        while attempts < self._max_retries:
            try:
                result = self._client.execute(
                    document=gql(query), variable_values=variables
                )
                return result
            except TransportQueryError as exc:
                errors = exc.errors
                if errors:
                    msg = errors[0]["message"]
                    if msg == "Operation only allowed for authenticated users.":
                        attempts += 1
                        if attempts == self._max_retries:
                            raise APIError(detail=msg) from exc
                        self._token_refresh()
                        time.sleep(self._retry_interval)
                        continue
                    raise APIError(detail=msg) from exc
                raise APIError(detail=str(exc)) from exc
            except TransportError as exc:
                attempts += 1
                if attempts == self._max_retries:
                    raise APIError(detail=str(exc)) from exc
                time.sleep(self._retry_interval)

        raise APIError("Request failed.")

    def _token_refresh(self) -> None:
        refresh_token = get_token(TokenType.REFRESH)
        if refresh_token is None:
            raise AuthorizationError(
                "Failed to refresh access token. Please login again."
            )

        injector = get_injector()
        url_provider = injector.get(URLProvider)
        refresh_url = url_provider.get_web_backend_uri("/api/auth/session/refresh")

        response = httpx.post(
            refresh_url,
            headers={
                **get_auth_header(token=refresh_token),
                "Content-Type": "application/json",
                "Accept": "application/json",
                "rid": "anti-csrf",
                "st-auth-mode": "header",
            },
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AuthenticationError(
                "Failed to refresh token. Please login again."
            ) from exc
        headers = response.headers
        access_token = headers["st-access-token"]
        refresh_token = headers["st-refresh-token"]
        update_token(TokenType.ACCESS, access_token)
        update_token(TokenType.REFRESH, refresh_token)
        # Initialize GQL client again with the updated auth headers.
        self._client = get_default_gql_client()
