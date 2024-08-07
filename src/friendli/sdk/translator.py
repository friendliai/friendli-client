# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Sdk exception translator."""

from __future__ import annotations

from functools import wraps
from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, TypeVar

from httpx import HTTPStatusError
from pydantic import ValidationError

from .exception import ApiError, AuthenticationError, RemoteSystemError, SdkError

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")


def translate_exception(func: Callable[P, R]) -> Callable[P, R]:
    """Translate exceptions."""

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except HTTPStatusError as e:
            # TODO(AJ): add more translations
            if e.response.status_code == HTTPStatus.UNAUTHORIZED:
                raise AuthenticationError from e

            detail = "Unknown API error"
            raise ApiError(detail, e.response.status_code) from e
        except ValidationError as e:
            # TODO(AJ): Change error message.
            #           Instruct user to update their SDK version
            detail = "Validation error occurred"
            raise SdkError(detail) from e
        except Exception as e:
            # TODO(AJ): add more specific exceptions
            detail = "Unknown error occurred"
            raise RemoteSystemError(detail) from e

    return _wrapper
