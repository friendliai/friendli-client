# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typer import Typer

from ..const import Panel
from ..context import AppContext, TyperAppContext

group = Typer()


@group.command(
    "version",
    help="Show Friendli Suite version information.",
    rich_help_panel=Panel.COMMON,
    context_settings={"command_sorting_key": 30},
)
def _version(ctx: TyperAppContext) -> None:
    from ..action.common_version import run

    with AppContext(ctx.obj) as app_ctx:
        run(app_ctx)
