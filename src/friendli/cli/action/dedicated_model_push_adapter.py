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
    base_model_id: str,
    model_name: str | None,
    project_id: str | None,
) -> None:
    """Login to Friendli Suite via token."""
    ctx.sdk.system.ping()

    if project_id is None:
        ctx.console.print("Please provide a project ID.")
        raise typer.Abort

    resp = ctx.sdk.model.push_adapter_model(
        model_path=model_path,
        base_model_id=base_model_id,
        project_id=project_id,
        model_name=model_name,
    )
    if not (res := resp.dedicated_model_push_adapter_complete):
        ctx.console.print("[error]Adapter model push failed[/]")
        raise typer.Abort

    ctx.console.print("[success]Adapter model pushed successfully.[/]")
    ctx.console.print(f"[info]✓[/] id: {res.adapter.id}")  # type: ignore
    ctx.console.print(f"[info]✓[/] name: {res.adapter.name}")  # type: ignore
