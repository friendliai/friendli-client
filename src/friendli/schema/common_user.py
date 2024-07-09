# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Common system schema."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003

from ._base import ModelBase


class WhoamiResponse(ModelBase):
    """Whoami information."""

    user_id: str
    user_name: str | None
    user_email: str
    session_created_at: datetime
