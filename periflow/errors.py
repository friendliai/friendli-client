# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow SDK errors."""

from __future__ import annotations

from typing import Any, List, Optional


class PeriFlowError(Exception):
    """PeriFlow exception base."""


class AuthenticationError(PeriFlowError):
    """Authentication failure error."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize AuthTokenNotFoundError."""
        super().__init__(f"Failed to authenticate: {detail}")


class AuthorizationError(PeriFlowError):
    """Not authorized to access."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize AuthorizationError."""
        super().__init__(f"Not authorized to access: {detail}")


class AuthTokenNotFoundError(PeriFlowError):
    """Auth token is not found."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize AuthTokenNotFoundError."""
        super().__init__(f"Auth token is not found: {detail}")


class LowServicePlanError(PeriFlowError):
    """The feature is not supported in your service plan."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize LowServicePlanError."""
        super().__init__(f"The feature is not supported in your service plan: {detail}")


class InvalidConfigError(PeriFlowError):
    """Invalid configuration provided."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize InvalidConfigError."""
        super().__init__(f"Invalid configuration provided: {detail}")


class InvalidAttributesError(PeriFlowError):
    """Checkpoint attributes are not valid."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize InvalidAttributesError."""
        super().__init__(f"Invalid checkpoint attributes: {detail}")


class InvalidPathError(PeriFlowError):
    """Invalid path provided."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize InvalidPathError."""
        super().__init__(f"Invalid path provided: {detail}")


class EntityTooLargeError(PeriFlowError):
    """Entity is too large."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize EntityTooLargeError."""
        super().__init__(f"Entity is too large: {detail}")


class NotFoundError(PeriFlowError):
    """Requested resource is not found."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize NotFoundError."""
        super().__init__(f"The resource is not found: {detail}")


class CheckpointConversionError(PeriFlowError):
    """Checkpoint conversion failure error."""

    def __init__(self, msg: str) -> None:
        """Initialize CheckpointConversionError."""
        super().__init__(f"Conversion failed: {msg}")


class NotSupportedCheckpointError(CheckpointConversionError):
    """Checkpoint is not supported."""

    def __init__(self, invalid_option: str, valid_options: List[Any]) -> None:
        """Initialize NotSupportedCheckpointError."""
        super().__init__(
            f"{invalid_option} is not supported. Please use one of {valid_options}."
        )


class TokenizerNotFoundError(PeriFlowError):
    """Cannot find PeriFlow-compatible tokenizer info."""

    def __init__(self, msg: Optional[str] = None) -> None:
        """Initialize TokenizerNotFoundError."""
        docs_link = (
            "https://huggingface.co/docs/transformers/main_classes/tokenizer#tokenizer"
        )
        default_msg = (
            "PeriFlow only supports Hugging Face 'fast' tokenizer. "
            f"Refer to {docs_link} to get more info."
        )
        super().__init__(msg or default_msg)


class RequestTimeoutError(PeriFlowError):
    """Request timeout error."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize NotFoundError."""
        super().__init__(f"Request timeout: {detail}")


class APIError(PeriFlowError):
    """PeriFlow API error."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize APIError."""
        super().__init__(f"API error: {detail}")


class SessionClosedError(PeriFlowError):
    """API Session is closed."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize SessionClosedError."""
        super().__init__(
            f"Cannot send requests to the API server because session is not opened: {detail}"
        )


class NotSupportedError(PeriFlowError):
    """Feature is not supported."""

    def __init__(self, detail: Optional[str] = None) -> None:
        """Initialize NotSupportedError."""
        super().__init__(f"The feature is not supported: {detail}")


class PeriFlowInternalError(PeriFlowError):
    """Internal error of PeriFlow."""

    def __init__(self, message: Optional[str] = None) -> None:
        """Initialize PeriFlowInternalError."""
        if message is None:
            message = "Please contact to FriendliAI"
        else:
            message = message.rstrip() + "\nPlease contant to FriendliAI"

        super().__init__(message)


class InvalidGenerationError(PeriFlowInternalError):
    """Invalid generation error."""

    def __init__(self, message: Optional[str] = None) -> None:
        """Initialize invalid generation error."""
        super().__init__(f"Invalid generation result: {message}")
