# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Deployment Schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Deployment(BaseModel):
    """V1 deployment schema."""

    id: Optional[str]
    name: Optional[str]
    artifactId: Optional[str]
    gpuType: Optional[str]
    numGpu: Optional[int]
    status: Optional[str]
    createdAt: Optional[str]
    updatedAt: Optional[str]
