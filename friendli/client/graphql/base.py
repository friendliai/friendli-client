# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli GQL Client Service."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from gql import Client, gql
from gql.transport.exceptions import TransportError, TransportQueryError
from gql.transport.httpx import HTTPXTransport

from friendli.auth import get_auth_header
from friendli.di.injector import get_injector
from friendli.errors import APIError
from friendli.utils.url import URLProvider


def get_default_gql_client() -> Client:
    """Get a default GraphQL client."""
    injector = get_injector()
    url_provider = injector.get(URLProvider)
    gql_transport = HTTPXTransport(
        url=url_provider.get_web_backend_uri("api/graphql"),
        headers=get_auth_header(),
    )
    gql_client = Client(transport=gql_transport, fetch_schema_from_transport=True)
    return gql_client


class GqlClient:
    """Base interface of graphql client to Friendli system."""

    def __init__(
        self, client: Client, max_retries: int = 3, retry_interval: float = 1
    ) -> None:
        """Initialize GqlClient."""
        self._client = client
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
                    raise APIError(detail=errors[0]["message"]) from exc
                raise APIError(detail=str(exc)) from exc
            except TransportError as exc:
                attempts += 1
                if attempts == self._max_retries:
                    raise APIError(detail=str(exc)) from exc
                time.sleep(self._retry_interval)

        raise APIError("Request failed.")
