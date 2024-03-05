# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Endpoint Schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from friendli.schema.resource.v1.common import Creator


class EndpointPhase(BaseModel):
    """Endpoint phase schema."""

    step: Optional[str] = None
    msg: Optional[str] = None
    desiredReplica: Optional[int] = None
    currReplica: Optional[int] = None


class EndpointAdvancedConfig(BaseModel):
    """Endpoint advanced config schema."""

    maxBatchSize: Optional[int] = None
    autoscalingMin: Optional[int] = None
    autoscalingMax: Optional[int] = None


class Endpoint(BaseModel):
    """Endpoint schema."""

    id: Optional[str]
    name: Optional[str]
    hfModelRepo: Optional[str]
    gpuType: Optional[str]
    numGpu: Optional[int]
    status: Optional[str]
    createdBy: Optional[Creator]
    endpointUrl: Optional[str]
    createdAt: Optional[datetime]
    updatedAt: Optional[datetime]
    phase: Optional[EndpointPhase] = None
    advancedConfig: Optional[EndpointAdvancedConfig] = None
