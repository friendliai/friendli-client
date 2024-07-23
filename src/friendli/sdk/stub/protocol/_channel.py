# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Websockets based stub implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from graphql import ExecutionResult
    from typing_extensions import TypeAlias

Event: TypeAlias = tuple[str, Optional["ExecutionResult"]]


class QueueCompletedError(Exception):
    """Raised when trying to put/get an item to a completed queue."""


class EventChannel:
    """Special queue used for each query waiting for server messages."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[
            tuple[str, ExecutionResult | None] | BaseException | None
        ] = asyncio.Queue()
        self._completed: bool = False
        self._exception: BaseException | None = None

    @property
    def completed(self) -> bool:
        """Check if the channel is completed."""
        return self._completed

    @property
    def exception(self) -> BaseException | None:
        """Get the exception if any."""
        return self._exception

    async def get(self) -> Event:
        """Get the next event from the queue."""
        if self._completed:
            raise QueueCompletedError

        item = await self._queue.get()
        self._queue.task_done()

        if isinstance(item, BaseException):
            self._completed = True
            self._exception = item
            raise item

        if item is None:
            self._completed = True
            raise QueueCompletedError

        answer_type, _ = item
        if answer_type == "complete":
            self._completed = True

        return item

    async def put(self, item: Event) -> None:
        if self._completed:
            raise QueueCompletedError

        await self._queue.put(item)

    async def set_exception(self, exception: BaseException) -> None:
        """Set the exception for this channel."""
        await self._queue.put(exception)
        self._completed = True

    async def complete(self) -> None:
        """Complete the channel."""
        await self._queue.put(None)
        self._completed = True
