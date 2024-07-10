# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from pathlib import Path

    from ..context import AppContext


def run(
    ctx: AppContext,
    model_path: Path,
    model_name: str | None,
    project_id: str | None,
) -> None:
    """Login to Friendli Suite via token."""
    # Handle ping error
    ctx.sdk.system.ping()
    if project_id is None:
        ctx.console.print("Please provide a project ID.")
        raise typer.Abort

    resp = ctx.sdk.model.push_base_model(
        model_path=model_path,
        project_id=project_id,
        model_name=model_name,
    )
    if not (res := resp.dedicated_model_push_base_complete):
        ctx.console.print("[error]Base model push failed[/]")
        raise typer.Abort

    ctx.console.print("[success]Base model pushed successfully.[/]")
    ctx.console.print(f"[info]✓[/] id: {res.model.id}")  # type: ignore
    ctx.console.print(f"[info]✓[/] name: {res.model.name}")  # type: ignore
