# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Common system schema."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class FileDescriptorInput(BaseModel):
    """File descriptor."""

    digest: str
    filename: str
    size: int
    path: Path
