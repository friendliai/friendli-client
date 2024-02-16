# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Context management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import friendli
from friendli.utils.fs import get_friendli_directory

team_context_path = get_friendli_directory() / "team"
project_context_path = get_friendli_directory() / "project"


def get_current_team_id() -> Optional[str]:
    """Get team ID of the current context."""
    if friendli.team_id:
        return friendli.team_id

    if not team_context_path.exists():
        return None

    with open(team_context_path, "r", encoding="utf-8") as f:
        group_id = f.read()
        return group_id


def set_current_team_id(team_id: str):
    """Set the current team context."""
    with open(team_context_path, "w", encoding="utf-8") as f:
        f.write(str(team_id))


def get_current_project_id() -> Optional[str]:
    """Get project ID of the current context."""
    if friendli.project_id:
        return friendli.project_id

    if not project_context_path.exists():
        return None

    with open(project_context_path, "r", encoding="utf-8") as f:
        project_id = f.read()
        return project_id


def set_current_project_id(pf_project_id: str):
    """Set the current project context."""
    with open(project_context_path, "w", encoding="utf-8") as f:
        f.write(str(pf_project_id))


def clear_contexts():
    """Clear all contexts."""
    Path.unlink(team_context_path)
    Path.unlink(project_context_path)
