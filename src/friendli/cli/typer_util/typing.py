# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Typer interfaces."""

from __future__ import annotations

from typing import Callable, Generic, TypedDict, TypeVar

import typer


class ContextSettings(TypedDict, total=False):
    """Context settings for a command."""

    # Click context settings
    # https://click.palletsprojects.com/en/8.1.x/api/#click.Context
    auto_envvar_prefix: str | None
    terminal_width: int | None
    max_content_width: int | None
    resilient_parsing: bool | None
    allow_extra_args: bool | None
    ignore_interspersed_args: bool | None
    ignore_unknown_options: bool | None
    help_option_names: list[str] | None
    token_normalize_func: Callable[[str], str] | None
    color: bool | None
    show_default: bool | None

    # Typer utils context settings
    command_sorting_key: int | None


T = TypeVar("T")  # Generic type variable for the context object


class TyperContext(typer.Context, Generic[T]):
    """Base typer context.

    Attributes:
        obj (T): generic type

    """

    obj: T
