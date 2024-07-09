# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Suite Sync SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Self

from ..graphql import GraphqlStub

if TYPE_CHECKING:
    from types import TracebackType

    from httpx import Client


class SyncClientBase:
    """Sync client Base."""

    def __init__(self, *, http_client: Client) -> None:
        """Initialize sync client."""
        self._client = http_client

    @property
    def http_client(self) -> Client:
        """Get HTTP client."""
        return self._client

    @property
    def gql_client(self) -> GraphqlStub:
        """Get GraphQL client."""
        return GraphqlStub(self._client, path="/api/graphql")

    def __enter__(self) -> Self:
        """Context manager for sync client."""
        self._client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self._client.__exit__(exc_type, exc_val, exc_tb)
