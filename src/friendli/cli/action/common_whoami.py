# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from ...sdk.exception import AuthenticationError
from ...util.humanize import humanize_datetime
from ..const import SUITE_PAT_URL

if TYPE_CHECKING:
    from ..context import AppContext


def run(ctx: AppContext) -> None:
    """Check currently logged in user."""
    ctx.sdk.system.ping()

    try:
        info = ctx.sdk.user.whoami()
    except AuthenticationError:
        ctx.console.print("Failed to login. Your token is invalid or expired.")
        ctx.console.print(
            f"Please create new token from [link={SUITE_PAT_URL}]Friendli Suite[/link]!"
        )
        raise typer.Exit(1) from None

    user_name = info.user_name or info.user_email
    created_at = humanize_datetime(info.session_created_at)

    ctx.console.print(f"[info]✓[/] Hi, {user_name}!")
    ctx.console.print(f"[info]✓[/] You are logged in with a token created {created_at}")
