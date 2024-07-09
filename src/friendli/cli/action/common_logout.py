# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import AppContext


def run(ctx: AppContext) -> None:
    """Log out."""
    if ctx.settings_backend.settings.user_info is None:
        ctx.console.print("[info]✓[/] You are already logged out")
        return

    user_id = ctx.settings_backend.settings.user_info.user_id
    ctx.auth_backend.clear_credential(user_id)
    ctx.settings_backend.logout()

    ctx.console.print("[success]You have been logged out.")
