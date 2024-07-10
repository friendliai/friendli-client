# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.table import Table

from ...util.humanize import humanize_datetime

if TYPE_CHECKING:

    from ..context import AppContext


def run(ctx: AppContext, project_id: str) -> None:
    """Login to Friendli Suite via token."""
    # Handle ping error
    ctx.sdk.system.ping()

    resp = ctx.sdk.model.list(project_id=project_id)

    total = resp.total_count

    table = Table(title=f"Total models in project: {total}")

    table.add_column("ID", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Created At", justify="right")

    for edge in resp.edges:
        node = edge.node
        created_at = humanize_datetime(node.created_at) if node.created_at else "N/A"
        table.add_row(node.id, node.name, created_at)

    ctx.console.print(table)
