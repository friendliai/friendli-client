# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Retry wrapper for httpx transport."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import closing
from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, Generator, Generic, Iterable, TypeVar, Union

from httpx import (
    AsyncBaseTransport,
    BaseTransport,
    NetworkError,
    ProtocolError,
    Request,
    Response,
)

from .backoff import generate_sequence
from .header import RetryAfterHeader

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

TransportT = TypeVar("TransportT", bound=Union[BaseTransport, AsyncBaseTransport])
SequenceT: "TypeAlias" = Generator[float, None, None]


def _default_backoff_sequence() -> SequenceT:
    return generate_sequence("exponential", base=0.1, jitter=True, cap=10)


class _BaseRetryTransportWrapper(Generic[TransportT]):
    """A HTTP transport wrapper that automatically retries requests.

    The transport will retry requests for specific HTTP status codes and request
    methods. The wait time between retries increases with each attempt, and a random
    amount of jitter is added to avoid a "thundering herd" effect.

    It will NOT retry requests that fail due to connection errors. For that, use the
    `retries` argument to `AsyncHTTPTransport` or `HTTPTransport`.

    The transport will also respect the Retry-After header in HTTP responses, if
    present.

    The wrapper will work with both sync and async transports.

    Args:
        transport (BaseTransport | AsyncBaseTransport): The underlying HTTP transport.
        status_codes (Iterable[int] | None): The HTTP status codes that can be retried.
            Defaults to 429, 502, 503, 504.
        http_methods (Iterable[str] | None): The HTTP methods that can be retried.
            Defaults to HEAD, GET, PUT, DELETE, OPTIONS, TRACE.
        max_attempts (int): The maximum number of times to retry a request.
            Defaults to 10.
        backoff_sequence_gen (Callable[[], Iterable[float]]): A function that returns an
            backoff sequence. Defaults to exponential backoff.
        respect_retry_after_header (bool): Whether to respect the Retry-After header
        max_retry_after (int): The maximum number of seconds to wait for a Retry-After

    """

    DEFAULT_HTTP_METHODS = frozenset(
        ["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    DEFAULT_RETRY_STATUS_CODES = frozenset(
        [
            HTTPStatus.TOO_MANY_REQUESTS,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.GATEWAY_TIMEOUT,
        ]
    )
    DEFAULT_MAX_ATTEMPTS = 10
    DEFAULT_RETRY_AFTER = 60

    def __init__(  # noqa: PLR0913
        self,
        transport: TransportT,
        *,
        status_codes: Iterable[int] = DEFAULT_RETRY_STATUS_CODES,
        http_methods: Iterable[str] = DEFAULT_HTTP_METHODS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_sequence_gen: Callable[[], SequenceT] = _default_backoff_sequence,
        respect_retry_after_header: bool = True,
        max_retry_after: int = DEFAULT_RETRY_AFTER,
        retry_on_connection_errors: bool = True,
    ) -> None:
        """Initialize."""
        if max_attempts < 1:
            msg = "max_attempts must be at least 1"
            raise ValueError(msg)
        # TODO: Warn user if max_attempts is too low

        self._transport = transport
        self._max_attempts = max_attempts
        self._backoff_sequence_gen = backoff_sequence_gen
        self._respect_retry_after_header = respect_retry_after_header

        self._http_methods = frozenset(http_methods)
        self._retry_status_codes = frozenset(status_codes)
        self._max_retry_after = max_retry_after
        self._retry_on_connection_errors = retry_on_connection_errors

    # log debug retry info
    def _backoff_retry_flow(self) -> Generator[float, Response, Response]:
        """Generate backoff times and handles retries.

        It defines the retry logic and is used by `handle_request` to send requests.

        """
        with closing(self._backoff_sequence_gen()) as seq:
            resp = yield 0

            for _ in range(self._max_attempts - 1):
                if resp.status_code not in self._retry_status_codes:
                    return resp

                retry_after_header = resp.headers.get("Retry-After")
                backoff_time: int | float
                if self._respect_retry_after_header and retry_after_header:
                    header = RetryAfterHeader.parse(retry_after_header)
                    backoff_time = float(
                        min(header.delta.seconds, self._max_retry_after)
                    )
                else:
                    backoff_time = next(seq)

                resp = yield backoff_time

            return resp


class RetryTransportWrapper(_BaseRetryTransportWrapper[BaseTransport], BaseTransport):
    """A HTTP transport wrapper that automatically retries requests."""

    def handle_request(self, request: Request) -> Response:
        """Sends an HTTP request, possibly with retries.

        Args:
            request (Request): The request to send.

        Returns:
            Response: The response received.

        """
        if request.method not in self._http_methods:
            return self._transport.handle_request(request)

        with closing(self._backoff_retry_flow()) as backoff:
            try:
                backoff_time = next(backoff)
                while True:
                    time.sleep(backoff_time)
                    resp = self._handle_request(request)
                    backoff_time = backoff.send(resp)
            except StopIteration as e:
                return e.value

    def close(self) -> None:
        """Closes the underlying HTTP transport, terminating all connections."""
        self._transport.close()

    def _handle_request(self, request: Request) -> Response:
        try:
            return self._transport.handle_request(request)
        except (ProtocolError, NetworkError) as e:
            logger.debug("Connection error: %s", e)
            if not self._retry_on_connection_errors:
                raise

        return self._transport.handle_request(request)


class AsyncRetryTransportWrapper(
    _BaseRetryTransportWrapper[AsyncBaseTransport], AsyncBaseTransport
):
    """A HTTP transport wrapper that automatically retries requests."""

    async def handle_async_request(self, request: Request) -> Response:
        """Sends an HTTP request, possibly with retries.

        Args:
            request: The request to perform.

        Returns:
            The response.

        """
        if request.method not in self._http_methods:
            return await self._transport.handle_async_request(request)

        with closing(self._backoff_retry_flow()) as backoff:
            try:
                backoff_time = next(backoff)
                while True:
                    await asyncio.sleep(backoff_time)
                    resp = await self._handle_request(request)
                    backoff_time = backoff.send(resp)
            except StopIteration as e:
                return e.value

    async def aclose(self) -> None:
        """Closes the underlying HTTP transport, terminating all connections."""
        await self._transport.aclose()

    async def _handle_request(self, request: Request) -> Response:
        try:
            return await self._transport.handle_async_request(request)
        except (ProtocolError, NetworkError) as e:
            logger.debug("Connection error: %s", e)
            if not self._retry_on_connection_errors:
                raise

        return await self._transport.handle_async_request(request)
