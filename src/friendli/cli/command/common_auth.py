# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich import print
from typer import Abort, Argument, FileText, Option, Typer

from ...util.email import validate_email
from ..const import Panel
from ..context import AppContext, TyperAppContext
from ..typer_util import CommandUsageExample, format_examples

group = Typer()


@group.command(
    "login",
    help="""
Login to Friendli Suite.

ㅤ
By default, you will be redirected to the Friendli Suite login page on your browser. \
You can also login using your password or personal access token directly on your \
terminal. DO NOT type your password or token directly in your terminal.
""",
    rich_help_panel=Panel.COMMON,
    epilog=format_examples(
        [
            CommandUsageExample(
                synopsis=(
                    "Use browser to login to Friendli Suite. [yellow](RECOMMENDED)[/]"
                ),
                args="friendli login",
            ),
            CommandUsageExample(
                synopsis="Use email and password to login.",
                args="friendli login person@john.doe --with-password < password.txt",
            ),
            CommandUsageExample(
                synopsis="Use personal access token to login.",
                args="echo -n $FRIENDLI_TOKEN | friendli login --with-token",
            ),
        ]
    ),
    context_settings={"command_sorting_key": 10},
)
def _login(
    ctx: TyperAppContext,
    email: Optional[str] = Argument(
        default=None,
        callback=validate_email,
        help="Your email address. [dim](Optional)[/dim]",
        show_default=False,
    ),
    *,
    with_web: bool = Option(  # noqa: ARG001
        True,  # noqa: FBT003
        "--with-web",
        show_default=False,
        help="Open browser to authenticate. [dim](Default)[/dim]",
        rich_help_panel="Authentication Options",
    ),
    with_password: bool = Option(
        False,  # noqa: FBT003
        "--with-password",
        show_default=False,
        help="Read password from standard input to authenticate.",
        rich_help_panel="Authentication Options",
    ),
    with_token: bool = Option(
        False,  # noqa: FBT003
        "--with-token",
        show_default=False,
        help="Read personal access token from standard input to authenticate.",
        rich_help_panel="Authentication Options",
    ),
    _stdin: FileText = Option(sys.stdin, hidden=True),  # noqa: B008
) -> None:
    if with_password:
        if email is None:
            print("You must provide an email address to authenticate with.")
            print("")
            print("  [cyan]$ friendli login person@john.doe --with-password[/]")
            print("")
            raise Abort

        print(
            (
                "Login with email password is not supported yet.\n"
                "Please use token to login.\n\n"
                "$ [cyan]echo -n $FRIENDLI_TOKEN | friendli login --with-token[/]"
            )
        )

        return

    if with_token:
        token = _stdin.read()
        from ..action.common_login_via_token import run as _login_via_token

        with AppContext(ctx.obj) as app_ctx:
            _login_via_token(app_ctx, token=token)

        return

    print(
        (
            "Login with browser is not supported yet.\n"
            "Please use token to login.\n\n"
            "$ [cyan]echo -n $FRIENDLI_TOKEN | friendli login --with-token[/]"
        )
    )
    raise typer.Abort


@group.command(
    "logout",
    help="Logout from Friendli Suite.",
    rich_help_panel=Panel.COMMON,
    context_settings={"command_sorting_key": 10},
)
def _logout(ctx: TyperAppContext) -> None:
    from ..action.common_logout import run

    with AppContext(ctx.obj) as app_ctx:
        run(app_ctx)


@group.command(
    "whoami",
    help="View account information of logged in user.",
    rich_help_panel=Panel.COMMON,
    context_settings={"command_sorting_key": 10},
)
def _whoami(ctx: TyperAppContext) -> None:
    from ..action.common_whoami import run

    with AppContext(ctx.obj) as app_ctx:
        run(app_ctx)
