# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Decorator utils."""

from __future__ import annotations

import functools
from typing import Any, Callable

from friendli.errors import APIError, AuthenticationError, AuthorizationError
from friendli.utils.format import secho_error_and_exit


def check_api(func: Callable[..., Any]) -> Callable[..., Any]:
    """Check common API exceptions."""

    @functools.wraps(func)
    def inner(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except (APIError, AuthorizationError, AuthenticationError) as exc:
            secho_error_and_exit(str(exc))

    return inner


def check_api_params(func: Callable[..., Any]) -> Callable[..., Any]:
    """Check API params."""

    @functools.wraps(func)
    def inner(*args, **kwargs) -> Any:
        model = kwargs["model"]
        endpoint_id = kwargs["endpoint_id"]

        if model is None and endpoint_id is None:
            secho_error_and_exit("One of 'model' and 'endpoint_id' should be provided.")
        if model is not None and endpoint_id is not None:
            secho_error_and_exit(
                "Only one of 'model' and 'endpoint_id' should be provided."
            )

        return func(*args, **kwargs)

    return inner
