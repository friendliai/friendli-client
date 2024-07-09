# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli SDK Exceptions."""

from __future__ import annotations

from http import HTTPStatus


class SdkError(Exception):
    """Friendli sdk exception base."""


class RemoteSystemError(SdkError):
    """Raised when remote system error occurs."""

    def __init__(self, detail: str) -> None:
        """Initialize SystemError."""
        msg = f"System error: {detail}"
        super().__init__(msg)


class ApiError(SdkError):
    """Raised when API requests returns unexpected status code."""

    def __init__(self, detail: str, status_code: int) -> None:
        """Initialize ApiError."""
        msg = f"API error: {detail} (status code: {status_code})"
        super().__init__(msg)


class AuthenticationError(ApiError):
    """Authentication failure error."""

    def __init__(self, detail: str | None = None) -> None:
        """Initialize AuthTokenNotFoundError."""
        detail = detail or "Authentication failed"
        super().__init__(detail, HTTPStatus.UNAUTHORIZED)
