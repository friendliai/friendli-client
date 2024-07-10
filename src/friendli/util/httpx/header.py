# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Models for parsing header values."""

from __future__ import annotations

import abc
from datetime import datetime, timedelta, timezone
from typing import TypeVar

T = TypeVar("T", bound="HeaderValue")


class HeaderValue(abc.ABC):
    """Header model."""

    @classmethod
    @abc.abstractmethod
    def parse(cls: type[T], value: str) -> T:
        """Parse header value.

        Args:
            value (str): header value

        Returns:
            HeaderValue: parsed header value

        Raises:
            ValueError: if value is invalid

        """

    @abc.abstractmethod
    def __str__(self) -> str:
        """Return original header value."""


class RetryAfterHeader(HeaderValue):
    """Retry-After header value."""

    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Date#syntax
    DateFormat = "%a, %d %b %Y %H:%M:%S GMT"

    def __init__(self, retry_time: datetime) -> None:
        """Initialize."""
        self._retry_time = retry_time

    @property
    def retry_time(self) -> datetime:
        """Return retry time."""
        return self._retry_time

    @property
    def delta(self) -> timedelta:
        """Return delta."""
        delta = self.retry_time - datetime.now(tz=timezone.utc)
        return max(delta, timedelta(0))

    @classmethod
    def parse(cls, value: str) -> RetryAfterHeader:
        """Parse header value."""
        if value.isdigit():
            seconds = int(value)
            retry_time = datetime.now(tz=timezone.utc) + timedelta(seconds=seconds)
            return cls(retry_time)

        retry_time = datetime.strptime(value, cls.DateFormat)  # noqa: DTZ007
        retry_time = retry_time.replace(tzinfo=timezone.utc)
        return cls(retry_time)

    def __str__(self) -> str:
        """Return original header value."""
        return self.retry_time.strftime(RetryAfterHeader.DateFormat)
