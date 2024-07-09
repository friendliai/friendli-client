# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Show client and server version."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...const import __version__

if TYPE_CHECKING:
    from ..context import AppContext


def run(ctx: AppContext) -> None:
    """Show client and server version."""
    msg = f"Friendli Client (SDK): [cyan bold]{__version__}[/]"
    ctx.console.print(msg)

    system_version = ctx.sdk.system.version()

    msg = f"Friendli Suite  (Server): [cyan bold]{system_version.version}[/]"
    ctx.console.print(msg)
