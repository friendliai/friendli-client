# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Decorator utils."""

from __future__ import annotations

import functools
from typing import Any, Callable

from friendli.errors import APIError, AuthorizationError
from friendli.utils.format import secho_error_and_exit


def check_api(func: Callable[..., Any]) -> Callable[..., Any]:
    """Check common API exceptions."""

    @functools.wraps(func)
    def inner(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except (APIError, AuthorizationError) as exc:
            secho_error_and_exit(str(exc))

    return inner
