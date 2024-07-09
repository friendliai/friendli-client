# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Websockets based stub implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing, suppress
from itertools import count
from typing import TYPE_CHECKING, Any, AsyncGenerator, Mapping

from graphql import ExecutionResult
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

from ..exception import GqlServerError, GqlServerProtocolError
from ..protocol._channel import EventChannel
from ..typing import AsyncStubInterface

if TYPE_CHECKING:
    from websockets.client import WebSocketClientProtocol


logger = logging.getLogger(__name__)


class AsyncWebsocketsStub(AsyncStubInterface):
    """Async stub with websockets backend."""

    GraphqlwsSubprotocol = Subprotocol("graphql-transport-ws")
    SupportedSubprotocols = (GraphqlwsSubprotocol,)

    PongRequestId = 0

    def __init__(self, client: WebSocketClientProtocol) -> None:
        """Initialize."""
        self._client = client

        self._query_id_gen = count(self.PongRequestId + 1)

        # First channel is for heartbeat messages
        self._channels: dict[int, EventChannel] = {}
        self._channels[self.PongRequestId] = EventChannel()

        self._bg_tasks: list[asyncio.Task[None]] = []

    async def aconnect(self) -> None:
        """Connect to the GraphQL server.

        Steps:
            1. Send init message
            2. Wait for connection acknowledge from the server
            3. Create an asyncio task which will be used to receive & parse responses

        """
        subprotocol = self._client.response_headers.get(
            "Sec-WebSocket-Protocol", self.GraphqlwsSubprotocol
        )

        # Send the init message and wait for the ack from the server
        init_message = json.dumps({"type": "connection_init", "payload": {}})
        await self._client.send(init_message)
        init_answer = str(await self._client.recv())
        answer_type, *_ = self._parse_answer(init_answer)
        if answer_type != "connection_ack":
            msg = f"Unexpected answer from the server: {init_answer}"
            raise GqlServerProtocolError(msg)

        # for graphqlws, send a ping message every 30 seconds
        if subprotocol == self.GraphqlwsSubprotocol:
            self._bg_tasks.append(asyncio.create_task(self._send_ping_coro()))

        self._bg_tasks.append(asyncio.create_task(self._receive_data_loop()))

    async def aclose(self) -> None:
        """Close the connection to the GraphQL server."""
        for task in self._bg_tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        for channel in self._channels.values():
            await channel.complete()
        self._channels.clear()

    async def __execute__(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> ExecutionResult:
        """Execute a GraphQL query."""
        async with aclosing(
            self.__subscribe__(query, variables, operation_name)
        ) as agen:
            results = [result async for result in agen]

        if len(results) == 0:
            msg = "No results received from the server"
            raise GqlServerProtocolError(msg)

        if len(results) > 1:
            msg = f"More than one result received from the server: {results}"
            raise GqlServerProtocolError(msg)

        return results[0]

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
        query_id = next(self._query_id_gen)
        channel = EventChannel()
        self._channels[query_id] = channel

        await self._send_query(query_id, query, variables, operation_name)

        try:
            while True:
                answer_type, execution_result = await channel.get()

                if execution_result is not None:
                    yield execution_result

                elif answer_type == "complete":
                    break
        except (GeneratorExit, asyncio.CancelledError):
            payload = {"id": str(query_id), "type": "complete"}
            await self._client.send(json.dumps(payload))

        finally:
            await channel.complete()
            del self._channels[query_id]

    async def _send_query(
        self,
        query_id: int,
        query: str,
        variable_values: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> None:
        """Send a query to the provided websocket connection.

        We use an incremented id to reference the query.

        Returns the used id for this query.
        """
        payload: dict[str, Any] = {"query": query}
        if variable_values:
            payload["variables"] = variable_values
        if operation_name:
            payload["operationName"] = operation_name

        query_type = "subscribe"

        payload = {"id": str(query_id), "type": query_type, "payload": payload}
        query_str = json.dumps(payload)
        await self._client.send(query_str)

    ############################
    # Keep Alive Health Checks #
    ############################

    async def _send_ping(self, payload: Any | None = None) -> None:  # noqa: ANN401
        """Send a ping message for the graphql-ws protocol."""
        ping_message = {"type": "ping"}

        if payload is not None:
            ping_message["payload"] = payload

        await self._client.send(json.dumps(ping_message))

    async def _send_pong(self, payload: Any | None = None) -> None:  # noqa: ANN401
        """Send a pong message for the graphql-ws protocol."""
        pong_message = {"type": "pong"}

        if payload is not None:
            pong_message["payload"] = payload

        await self._client.send(json.dumps(pong_message))

    async def _send_ping_coro(self) -> None:
        """Coroutine to periodically send a ping from the client to the backend."""
        ping_interval = 10
        pong_timeout = 20
        event_channel = self._channels[self.PongRequestId]

        try:
            while True:
                await asyncio.sleep(ping_interval)
                await self._send_ping()
                await asyncio.wait_for(event_channel.get(), pong_timeout)
        except asyncio.TimeoutError:
            await self.aclose()

    ############################
    # Receive & Parse Messages #
    ############################

    async def _receive_data_loop(self) -> None:
        """Coroutine to receive & parse messages from the server."""
        while True:
            answer = str(await self._client.recv())

            try:
                answer_type, answer_id, execution_result = self._parse_answer(answer)
                logger.debug(
                    "Received answer: %s %s %s",
                    answer_type,
                    answer_id,
                    execution_result,
                )
            except ConnectionClosed:
                await self.aclose()
                return

            await self._handle_answer(answer_type, answer_id, execution_result)

    def _parse_answer(
        self, answer: str
    ) -> tuple[str, int | None, ExecutionResult | BaseException | None]:
        """Parse the answer received from the server.

        Returns:
            - the answer_type
                ('connection_ack', 'ping', 'pong', 'data', 'error', 'complete')
            - the answer id (Integer) if received or None
            - an execution Result if the answer_type is 'data' or None

        """
        json_answer: Mapping[str, Any] = json.loads(answer)

        answer_type: str | None = json_answer.get("type")
        if answer_type is None:
            msg = "answer_type not found"
            raise GqlServerProtocolError(msg)

        if answer_type in ("ping", "pong", "connection_ack"):
            return answer_type, None, None

        if answer_type not in ("next", "error", "complete"):
            msg = f"Unknown answer_type: {answer_type}"
            raise GqlServerProtocolError(msg)

        answer_id = int(json_answer["id"]) if "id" in json_answer else None
        if answer_type == "complete":
            return answer_type, answer_id, None

        payload = json_answer.get("payload")
        if answer_type == "next":
            if not isinstance(payload, dict):
                msg = "payload is not a dict"
                raise GqlServerProtocolError(msg)

            if "errors" not in payload and "data" not in payload:
                msg = "payload does not contain 'data' or 'errors' fields"
                raise GqlServerProtocolError(msg)

            execution_result = ExecutionResult(
                errors=payload.get("errors"),
                data=payload.get("data"),
                extensions=payload.get("extensions"),
            )
            return answer_type, answer_id, execution_result

        # answer_type is "error"
        if not isinstance(payload, list):
            msg = "error is not a list"
            raise GqlServerProtocolError(msg)

        return answer_type, answer_id, GqlServerError(payload)

    async def _handle_answer(
        self,
        answer_type: str,
        answer_id: int | None,
        execution_result: ExecutionResult | BaseException | None,
    ) -> None:
        if answer_type == "ping":
            await self._send_pong()
            return

        if answer_type == "pong":
            answer_id = 0

        if answer_id is None:
            return

        with suppress(KeyError):
            listener = self._channels[answer_id]

        if isinstance(execution_result, BaseException):
            await listener.set_exception(execution_result)
        else:
            await listener.put((answer_type, execution_result))
