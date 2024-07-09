# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Common system schema."""

from __future__ import annotations

from ._base import ModelBase


class VersionResponse(ModelBase):
    """System version."""

    version: str
