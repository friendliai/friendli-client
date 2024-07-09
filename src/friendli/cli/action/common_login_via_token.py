# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from ...sdk.exception import AuthenticationError
from ...util.humanize import humanize_datetime
from ..backend.settings import ApplicationConfig, UserInfo
from ..const import SUITE_PAT_URL

if TYPE_CHECKING:
    from ..context import AppContext


def run(ctx: AppContext, token: str) -> None:
    """Login to Friendli Suite via token."""
    ctx.sdk.system.ping()
    ctx.refresh_client(auth=token)

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

    ctx.console.print(f"[success]🎉 Welcome, {user_name}!\n")
    ctx.console.print(f"[info]✓[/] You are logged in with a token created {created_at}")

    # TODO(AJ): handle cases where pat is not set in keychain
    ctx.auth_backend.store_credential(info.user_id, token)
    settings = ApplicationConfig(
        user_info=UserInfo(user_id=info.user_id, auth_strategy="pat", context=None)
    )
    ctx.settings_backend.save(settings)
    ctx.console.print(
        "[info]✓[/] Your personal access token is stored safely in your keychain."
    )

    resp = ctx.sdk.team.list()
    if (data := resp.client_user) is None or (teams := data.teams) is None:
        ctx.console.print("[warn]🔔 No teams found.[/]")
        ctx.console.print("   Please contact us at support@friendli.ai for assistance.")

    team_count = teams.total_count
    ctx.console.print(f"[info]✓[/] {team_count} teams found.")

    # Show user that context is set on default team
    # Show user that user has access on dedicated endpoints
    # Show user that context is set on project

    # TODO(AJ): avoid project that user has only read only access to

    # Redirect user to docs page explaining default team (and/or context)

    # TODO(AJ): warn user about token expiration, nudge user to use other auth method
    #           show currently chosen team / project context
