# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typer import Typer

from ..const import Panel
from ..typer_util import CommandUsageExample, format_examples

app = Typer(
    no_args_is_help=True,
    name="team",
    rich_help_panel=Panel.DEDICATED,
    help="Manage your Friendli Team.",
    context_settings={"command_sorting_key": 20},
)


@app.command(
    "list",
    help="""
List all teams in your Friendli account.
""",
    epilog=format_examples(
        [
            CommandUsageExample(
                synopsis=(
                    "Use browser to login to Friendli Suite. [yellow](RECOMMENDED)[/]"
                ),
                args="login",
            ),
        ]
    ),
)
def _list() -> None:
    pass


@app.command(
    "set",
    help="""
List all teams in your Friendli account.
""",
    epilog=format_examples(
        [
            CommandUsageExample(
                synopsis=(
                    "Use browser to login to Friendli Suite. [yellow](RECOMMENDED)[/]"
                ),
                args="login",
            ),
        ]
    ),
)
def _set() -> None:
    pass
