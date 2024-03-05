# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Commonly used Schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Creator(BaseModel):
    """Fine-tuning creator schema."""

    id: Optional[str]
    name: Optional[str]
    email: Optional[str]
