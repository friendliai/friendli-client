# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Click context."""

from __future__ import annotations

from typing import Any, Callable, MutableMapping

from click.core import BaseCommand, Command, Context


class ExtendedContext(Context):
    """Extended context for Click context."""

    def __init__(  # noqa: PLR0913
        self,
        command: Command,
        parent: Context | None = None,
        info_name: str | None = None,
        obj: Any | None = None,  # noqa: ANN401
        auto_envvar_prefix: str | None = None,
        default_map: MutableMapping[str, Any] | None = None,
        terminal_width: int | None = None,
        max_content_width: int | None = None,
        resilient_parsing: bool = False,  # noqa: FBT001, FBT002
        allow_extra_args: bool | None = None,
        allow_interspersed_args: bool | None = None,
        ignore_unknown_options: bool | None = None,
        help_option_names: list[str] | None = None,
        token_normalize_func: Callable[[str], str] | None = None,
        color: bool | None = None,
        show_default: bool | None = None,
        command_sorting_key: int | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(
            command,
            parent,
            info_name,
            obj,
            auto_envvar_prefix,
            default_map,
            terminal_width,
            max_content_width,
            resilient_parsing,
            allow_extra_args,
            allow_interspersed_args,
            ignore_unknown_options,
            help_option_names,
            token_normalize_func,
            color,
            show_default,
        )
        self.command_sorting_key = command_sorting_key


BaseCommand.context_class = ExtendedContext
