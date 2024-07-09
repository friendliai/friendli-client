# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Typer utilities."""


from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from typer.core import TyperGroup

from ..const import Panel

if TYPE_CHECKING:
    from typing import MutableMapping

    from click import Context as ClickContext
    from typer import Typer
    from typer.core import TyperCommand


_PanelOrder = {p: i for i, p in enumerate(Panel)}


class OrderedCommands(TyperGroup):
    """Typer group that returns list of commands in the order of definition."""

    commands: MutableMapping[str, TyperCommand]  # type: ignore[assignment]

    def list_commands(self, _: ClickContext) -> list[str]:
        """Return list of commands in the order appear."""
        commands = sorted(
            self.commands.values(),
            key=lambda cmd: (
                _PanelOrder.get(cmd.rich_help_panel, sys.maxsize),  # type: ignore
                cmd.context_settings.get("command_sorting_key", 0),
            ),
        )
        return [cmd.name or "" for cmd in commands]


def merge_typer(base_app: Typer, app: Typer) -> None:
    """Merge typer app into base app."""
    base_app.registered_groups.extend(app.registered_groups)
    base_app.registered_commands.extend(app.registered_commands)
