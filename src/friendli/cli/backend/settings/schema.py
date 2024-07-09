# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""CLI application schema."""

from __future__ import annotations

from pydantic import BaseModel
from typing_extensions import Literal


class WorkingContext(BaseModel):
    """User's working context."""

    team_id: str
    project_id: str | None


class UserInfo(BaseModel):
    """User's login context."""

    user_id: str
    auth_strategy: Literal["pat"]
    context: WorkingContext | None


class ApplicationConfig(BaseModel):
    """Application configuration."""

    user_info: UserInfo | None
