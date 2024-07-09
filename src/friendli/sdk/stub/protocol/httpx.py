# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Httpx backed stub implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator, Mapping

from ..typing import AsyncStubInterface, SyncStubInterface
from ._mixin import AsyncRequestsLikeMixin, SyncRequestsLikeMixin

if TYPE_CHECKING:
    from graphql import ExecutionResult
    from httpx import AsyncClient, Client


class AsyncHttpxStub(AsyncRequestsLikeMixin, AsyncStubInterface):
    """Sync stub with httpx backend."""

    def __init__(self, client: AsyncClient, *, path: str = "") -> None:
        """Initialize stub."""
        super().__init__()
        self._client = client
        self._path = path

    async def aconnect(self) -> None:
        """Connect to the GraphQL server."""

    async def aclose(self) -> None:
        """Close the connection to the GraphQL server."""

    async def __subscribe__(  # type: ignore[override]
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> AsyncGenerator[ExecutionResult, None]:
        """Send a query and receive the results using a python async generator.

        The query can be a graphql query, mutation or subscription.

        The results are sent as an ExecutionResult object.
        """
        raise NotImplementedError

    async def _send_request(self, **kw: Any) -> Mapping[str, Any]:
        """Send a request to the GraphQL server."""
        resp = await self._client.post(self._path, **kw)
        resp.raise_for_status()
        # TODO: handle errors
        return resp.json()


class SyncHttpxStub(SyncRequestsLikeMixin, SyncStubInterface):
    """Sync stub with httpx backend."""

    def __init__(self, client: Client, *, path: str = "") -> None:
        """Initialize stub."""
        super().__init__()
        self._client = client
        self._path = path

    def connect(self) -> None:
        """Connect to the GraphQL server."""

    def close(self) -> None:
        """Close the connection to the GraphQL server."""

    def __subscribe__(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> Generator[ExecutionResult, None, None]:
        """Send a query and receive the results using a python async generator.

        The query can be a graphql query, mutation or subscription.

        The results are sent as an ExecutionResult object.
        """
        raise NotImplementedError

    def _send_request(self, **kw: Any) -> Mapping[str, Any]:
        """Send a request to the GraphQL server."""
        resp = self._client.post(self._path, **kw)
        resp.raise_for_status()
        # TODO: handle errors
        return resp.json()
