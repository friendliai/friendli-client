# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""GraphQL stub interface."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator, Mapping

if TYPE_CHECKING:
    from graphql import ExecutionResult
    from typing_extensions import Self


class AsyncStubInterface(abc.ABC):
    """Sans IO style async GraphQL stub."""

    @abc.abstractmethod
    async def aconnect(self) -> None:
        """Connect to the GraphQL server."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Close the connection to the GraphQL server."""

    async def ping(self) -> None:
        """Ping the GraphQL server."""
        await self.__execute__(query="{ __typename }")

    @abc.abstractmethod
    async def __execute__(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> ExecutionResult:
        """Execute a GraphQL query."""

    @abc.abstractmethod
    async def __subscribe__(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> AsyncGenerator[ExecutionResult, None]:
        """Send a query and receive the results using a python async generator.

        The query can be a graphql query, mutation or subscription.

        The results are sent as an ExecutionResult object.
        """
        if False:
            yield

    async def __aenter__(self: Self) -> Self:
        """Connect to the GraphQL server."""
        await self.aconnect()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the connection to the GraphQL server."""
        await self.aclose()


class SyncStubInterface(abc.ABC):
    """Sans IO style GraphQL stub."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Connect to the GraphQL server."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the connection to the GraphQL server."""

    def ping(self) -> None:
        """Ping the GraphQL server."""
        self.__execute__(query="{ __typename }")

    @abc.abstractmethod
    def __execute__(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> ExecutionResult:
        """Execute a GraphQL query."""

    @abc.abstractmethod
    def __subscribe__(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> Generator[ExecutionResult, None, None]:
        """Send a query and receive the results using a python generator.

        The query can be a graphql query, mutation or subscription.

        The results are sent as an ExecutionResult object.
        """
        if False:
            yield

    def __enter__(self: Self) -> Self:
        """Connect to the GraphQL server."""
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection to the GraphQL server."""
        self.close()
