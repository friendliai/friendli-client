# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""A complete stub implementation of the graphql-core library.

It mixes the use of httpx and websockets to provide a complete implementation
"""


from __future__ import annotations

import abc
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, TypeVar

from pydantic import BaseModel

from .types import UploadFile

if TYPE_CHECKING:
    from graphql import ExecutionResult
    from typing_extensions import TypeAlias

R = TypeVar("R", bound=BaseModel)

VariablesT: TypeAlias = Dict[str, Any]


class ExecutionError(Exception):
    """Execution error."""

    def __init__(self, result: ExecutionResult) -> None:
        """Init."""
        super().__init__()
        self.result = result


class AbstractSyncStub(abc.ABC):
    """Base GQL backed graphql client stub."""

    @abc.abstractmethod
    def __execute__(
        self, query: str, variables: VariablesT | None = None
    ) -> ExecutionResult:
        """Execute query.

        Args:
            query (str): The query to execute
            variables (VariablesT | None): The variables to use for the query

        Returns:
            ExecutionResult: The result of the query

        """

    @abc.abstractmethod
    def __subscribe__(
        self, query: str, variables: VariablesT | None = None
    ) -> Generator[ExecutionResult, None, None]:
        """Subscribe to query.

        Args:
            query (str): The query to subscribe to
            variables (VariablesT | None): The variables to use for the query

        Returns:
            Generator[ExecutionResult, None, None]: The result of the query

        """
        if False:
            yield

    def _query(
        self,
        gql: str,
        /,
        *,
        response_model: type[R],
        variables: BaseModel | None = None,
    ) -> R:
        """Run a query against the server.

        Args:
            gql (str): The query to run
            response_model (type[R]): The pydantic model to use to deserialize
                the response
            variables (BaseModel | None): The variables to use for the query

        Raises:
            ExecutionError: If the query failed

        Returns:
            The response from the server

        """
        return self._handle_query_or_mutation(gql, response_model, variables)

    def _mutation(
        self,
        gql: str,
        /,
        *,
        response_model: type[R],
        variables: BaseModel | None = None,
    ) -> R:
        """Run a mutation against the server.

        Args:
            gql (str): The mutation to run
            response_model (type[R]): The pydantic model to use to deserialize
                the response
            variables (BaseModel | None): The variables to use for the mutation

        Raises:
            ExecutionError: If the query failed

        Returns:
            The response from the server

        """
        return self._handle_query_or_mutation(gql, response_model, variables)

    def _subscription(
        self,
        gql: str,
        /,
        *,
        response_model: type[R],
        variables: BaseModel | None = None,
    ) -> Generator[R, None, None]:
        """Run a subscription against the server.

        Args:
            gql (str): The subscription to run
            response_model (type[R]): The pydantic model to use to deserialize
                the response
            variables (BaseModel | None): The variables to use for the subscription

        Raises:
            ExecutionError: If the query failed

        Returns:
            An iterable of the response from the server

        """
        variable_values = _serialize_variables(variables)

        gen = self.__subscribe__(gql, variables=variable_values)
        try:
            for result in gen:
                if result.errors:
                    raise ExecutionError(result)

                yield response_model.parse_obj(result)
        finally:
            gen.close()

    def _handle_query_or_mutation(
        self, gql: str, response_model: type[R], variables: BaseModel | None
    ) -> R:
        variable_values = _serialize_variables(variables)
        result = self.__execute__(gql, variables=variable_values)
        if result.errors:
            raise ExecutionError(result)
        return response_model.parse_obj(result.data)


class AbstractAsyncStub(abc.ABC):
    """Base GQL backed graphql client stub."""

    @abc.abstractmethod
    async def __execute__(
        self, query: str, variables: VariablesT | None = None
    ) -> ExecutionResult:
        """Execute query.

        Args:
            query (str): The query to execute
            variables (VariablesT | None): The variables to use for the query

        Returns:
            ExecutionResult: The result of the query

        """

    @abc.abstractmethod
    async def __subscribe__(
        self, query: str, variables: VariablesT | None = None
    ) -> AsyncGenerator[ExecutionResult, None]:
        """Subscribe to query.

        Args:
            query (str): The query to subscribe to
            variables (VariablesT | None): The variables to use for the query

        Returns:
            AsyncGenerator[ExecutionResult, None, None]: The result of the query

        """
        if False:
            yield

    async def _query(
        self,
        gql: str,
        /,
        *,
        response_model: type[R],
        variables: BaseModel | None = None,
    ) -> R:
        """Run a query against the server.

        Args:
            gql (str): The query to run
            response_model (type[R]): The pydantic model to use to deserialize
                the response
            variables (BaseModel | None): The variables to use for the query

        Raises:
            ExecutionError: If the query failed

        Returns:
            The response from the server

        """
        return await self._handle_query_or_mutation(gql, response_model, variables)

    async def _mutation(
        self,
        gql: str,
        /,
        *,
        response_model: type[R],
        variables: BaseModel | None = None,
    ) -> R:
        """Run a mutation against the server.

        Args:
            gql (str): The mutation to run
            response_model (type[R]): The pydantic model to use to deserialize
                the response
            variables (BaseModel | None): The variables to use for the mutation

        Raises:
            ExecutionError: If the query failed

        Returns:
            The response from the server

        """
        return await self._handle_query_or_mutation(gql, response_model, variables)

    async def _subscription(
        self,
        gql: str,
        /,
        *,
        response_model: type[R],
        variables: BaseModel | None = None,
    ) -> AsyncGenerator[R, None]:
        """Run a subscription against the server.

        Args:
            gql (str): The subscription to run
            response_model (type[R]): The pydantic model to use to deserialize
                the response
            variables (BaseModel | None): The variables to use for the subscription

        Raises:
            ExecutionError: If the query failed

        Returns:
            An async generator of the response from the server

        """
        variable_values = _serialize_variables(variables)

        gen = self.__subscribe__(gql, variables=variable_values)
        try:
            async for result in gen:
                if result.errors:
                    raise ExecutionError(result)

                yield response_model.parse_obj(result)
        finally:
            await gen.aclose()

    async def _handle_query_or_mutation(
        self, gql: str, response_model: type[R], variables: BaseModel | None
    ) -> R:
        variable_values = _serialize_variables(variables)
        result = await self.__execute__(gql, variables=variable_values)
        if result.errors:
            raise ExecutionError(result)

        return response_model.parse_obj(result.data)


def _serialize_variables(variables: BaseModel | None) -> VariablesT | None:
    """Serialize variables.

    Use fastapi's jsonable encoder to turn it into a json-compatible dict.
    Note) we transform UploadFile into `io.BytesIO`.

    """
    if variables is None:
        return None

    variables = variables.model_copy(deep=True)
    _inplace_update_files(variables)

    custom_encoder = {io.BytesIO: lambda v: v}
    _ = custom_encoder

    # TODO(AJ): handle custom encoder of bytesio
    return variables.model_dump(mode="json", by_alias=True) if variables else None


def _inplace_update_files(obj: BaseModel) -> None:
    """Find and extract files from the variables."""
    fields = {name: getattr(obj, name) for name in obj.__fields__}
    for field_name, field_value in fields.items():
        if isinstance(field_value, UploadFile):
            file = field_value.file
            stream = file.open("rb") if isinstance(file, Path) else io.BytesIO(file)
            setattr(obj, field_name, stream)

        elif isinstance(field_value, BaseModel):
            _inplace_update_files(field_value)
