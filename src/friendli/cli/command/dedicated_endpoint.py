# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typer import Typer

from ..const import Panel
from ..typer_util import CommandUsageExample, format_examples

app = Typer(
    no_args_is_help=True,
    name="endpoint",
    rich_help_panel=Panel.DEDICATED,
    help="Manage your Dedicated Endpoints.",
)


@app.command(
    "list",
    help="""
Login to Friendli Suite.

ㅤ
By default, you will be redirected to the Friendli Suite login page on your browser. \
You can also login using your password or personal access token directly on your \
terminal. DO NOT type your password or token directly in your terminal.
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
