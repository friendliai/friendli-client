# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Exceptions."""

from __future__ import annotations


class BaseGqlStubException(Exception):
    """Base exception for all exceptions in this module."""


class GqlServerConnectionError(BaseGqlStubException):
    """Raised when the server connection fails."""


class GqlServerProtocolError(BaseGqlStubException):
    """Raised when the server protocol is not respected."""


class GqlServerError(BaseGqlStubException):
    """Raised when the server protocol is not respected."""

    def __init__(self, errors: list) -> None:
        """Initialize the exception."""
        super().__init__(errors)
        self.errors = errors
