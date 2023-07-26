# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Serving API Interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)
from urllib.parse import urljoin

import aiohttp
import requests
from pydantic import BaseModel

from periflow.auth import get_auth_header
from periflow.client.deployment import DeploymentClient
from periflow.enums import DeploymentSecurityLevel
from periflow.errors import InvalidConfigError, NotFoundError
from periflow.logging import logger
from periflow.utils.request import DEFAULT_REQ_TIMEOUT
from periflow.utils.url import get_baseurl

_Generation = TypeVar("_Generation", bound=BaseModel)
_GenerationLine = TypeVar("_GenerationLine", bound=BaseModel)
_Options = TypeVar("_Options")


class GenerationStream(ABC, Generic[_GenerationLine, _Generation]):
    """Generation stream."""

    def __init__(self, response: requests.Response) -> None:
        """Initializes generation stream."""
        self._iter = response.iter_lines()

    def __iter__(self) -> GenerationStream:  # noqa: D105
        return self

    @abstractmethod
    def __next__(self) -> _GenerationLine:
        """Iterates the stream."""

    @abstractmethod
    def wait(self) -> Optional[_Generation]:
        """Waits for the generation to complete."""


class AsyncGenerationStream(ABC, Generic[_GenerationLine, _Generation]):
    """Asynchronous generation stream."""

    def __init__(self, response: aiohttp.ClientResponse) -> None:
        """Initializes generation stream."""
        self._iter = response.content.__aiter__()

    def __aiter__(self) -> AsyncGenerationStream:  # noqa: D105
        return self

    @abstractmethod
    async def __anext__(self) -> _GenerationLine:
        """Iterates the stream."""

    @abstractmethod
    async def wait(self) -> Optional[_Generation]:
        """Waits for the generation to complete."""


_GenerationStream = TypeVar("_GenerationStream", bound=GenerationStream)
_AsyncGenerationStream = TypeVar("_AsyncGenerationStream", bound=AsyncGenerationStream)


class ServingAPI(
    ABC, Generic[_Generation, _GenerationStream, _AsyncGenerationStream, _Options]
):
    """Serving API interface."""

    def __init__(
        self,
        deployment_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_security_level: Optional[DeploymentSecurityLevel] = None,
    ) -> None:
        """Initializes ServingAPI."""
        if deployment_id is None and endpoint is None:
            raise InvalidConfigError(
                "One of 'deployment_id' and 'endpoint' should be provided."
            )
        if deployment_id is not None and endpoint is not None:
            raise InvalidConfigError(
                "Only provide one between 'deployment_id' and 'endpoint'."
            )

        if deployment_id is not None:
            client = DeploymentClient()
            deployment = client.get_deployment(deployment_id)
            endpoint = deployment["endpoint"]
            if not endpoint:
                raise NotFoundError("Active endpoint for the deployment is not found.")
            self._endpoint = urljoin(
                get_baseurl(endpoint), os.path.join(deployment_id, self._api_path)
            )
            self._auth_required = deployment["config"]["infrequest_perm_check"]
        elif endpoint is not None:
            if deployment_security_level is None:
                raise InvalidConfigError(
                    "'deployment_security_level' should be provided."
                )
            self._endpoint = endpoint
            self._auth_required = (
                deployment_security_level == DeploymentSecurityLevel.PROTECTED
            )

        self._session: Optional[aiohttp.ClientSession] = None

    def _get_headers(self) -> Dict[str, Any]:
        if self._auth_required:
            return get_auth_header()
        return {}

    @property
    @abstractmethod
    def _api_path(self) -> str:
        """Serving API URL path."""

    @overload
    @abstractmethod
    def create(self, options: _Options, *, stream: Literal[True]) -> _GenerationStream:
        ...

    @overload
    @abstractmethod
    def create(self, options: _Options, *, stream: Literal[False]) -> _Generation:
        ...

    @abstractmethod
    def create(
        self, options: _Options, *, stream: bool = False
    ) -> Union[_GenerationStream, _Generation]:
        """Creates a new serving result."""

    @overload
    @abstractmethod
    async def acreate(
        self, options: _Options, *, stream: Literal[True]
    ) -> _AsyncGenerationStream:
        ...

    @overload
    @abstractmethod
    async def acreate(
        self, options: _Options, *, stream: Literal[False]
    ) -> _Generation:
        ...

    @abstractmethod
    async def acreate(
        self, options: _Options, *, stream: bool = False
    ) -> Union[_AsyncGenerationStream, _Generation]:
        """Async API to create a new serving result."""

    @asynccontextmanager
    async def api_session(self) -> AsyncIterator[None]:
        """Creates a new API session."""
        if self._session is not None:
            logger.warn("API session is already opened.")
            return

        timeout = aiohttp.ClientTimeout(total=DEFAULT_REQ_TIMEOUT)
        async with aiohttp.ClientSession(
            headers=self._get_headers(), timeout=timeout
        ) as session:
            self._session = session
            try:
                yield
            finally:
                self._session = None
