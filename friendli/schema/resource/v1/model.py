# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Model Schemas."""

from __future__ import annotations

from pydantic import BaseModel


class Model(BaseModel):
    """Model schema."""

    name: str
    id: str
