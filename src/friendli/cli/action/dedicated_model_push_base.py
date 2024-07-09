# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    _ = model_path, model_name, project_id
