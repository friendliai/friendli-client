# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Context (Organization / Project) managing."""

from __future__ import annotations

import uuid
from typing import Optional

import friendli
from friendli.utils.fs import get_friendli_directory

org_context_path = get_friendli_directory() / "organization"
project_context_path = get_friendli_directory() / "project"


def get_current_group_id() -> Optional[uuid.UUID]:
    """Get organization ID of the current context."""
    if friendli.org_id:
        try:
            return uuid.UUID(friendli.org_id)
        except ValueError as exc:
            raise ValueError("Invalid organization ID format") from exc

    if not org_context_path.exists():
        return None

    with open(org_context_path, "r", encoding="utf-8") as f:
        group_id = uuid.UUID(f.read())
        return group_id


def set_current_group_id(pf_group_id: uuid.UUID):
    """Set the current organization context."""
    with open(org_context_path, "w", encoding="utf-8") as f:
        f.write(str(pf_group_id))


def get_current_project_id() -> Optional[uuid.UUID]:
    """Get project ID of the current context."""
    if friendli.project_id:
        try:
            return uuid.UUID(friendli.project_id)
        except ValueError as exc:
            raise ValueError("Invalid project ID format") from exc

    if not project_context_path.exists():
        return None

    with open(project_context_path, "r", encoding="utf-8") as f:
        project_id = uuid.UUID(f.read())
        return project_id


def set_current_project_id(pf_project_id: uuid.UUID):
    """Set the current project context."""
    with open(project_context_path, "w", encoding="utf-8") as f:
        f.write(str(pf_project_id))
