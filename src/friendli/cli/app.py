# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli CLI application definition."""

from __future__ import annotations

import warnings
from typing import Optional

from rich import print
from typer import Context, Option, Typer

from .command import app as command_app
from .const import Panel
from .context import RootContextObj, TyperAppContext
from .typer_util import OrderedCommands, merge_typer, run_trogon

warnings.filterwarnings("ignore")


# TODO(AJ): show on subcommands also
def _app_callback(
    ctx: TyperAppContext,
    token: Optional[str] = Option(None, "--token", help="Login token"),
    base_url: Optional[str] = Option(
        None,
        "--base-url",
        help="API URL",
        envvar="FRIENDLI_BASE_URL",
        hidden=True,
    ),
) -> None:
    if ctx.resilient_parsing:
        # Called when autocomplete is enabled
        return

    if base_url is not None:
        msg = f"[magenta]🔔 Heads up! You're using a custom URL:[/] {base_url}\n"
        print(msg)

    obj = RootContextObj(base_url=base_url, token=token)
    ctx.obj = obj


app = Typer(
    cls=OrderedCommands,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    # TODO(AJ): review add_completion=False
    add_completion=False,
    rich_markup_mode="rich",
    callback=_app_callback,
)

# TODO(AJ): add example about token
# -t TOKEN, --token=TOKEN

# Merge commands
merge_typer(app, command_app)


@app.command(
    name="tui",
    help="Open interactive command viewer.",
    rich_help_panel=Panel.OTHER,
)
def _interactive_help(ctx: Context) -> None:
    run_trogon(app, ctx, ignore_names=["tui"])
