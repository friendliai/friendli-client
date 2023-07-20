# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint Schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from periflow.enums import (
    CatalogStatus,
    CheckpointCategory,
    CheckpointStatus,
    CheckpointValidationStatus,
)


class V1Catalog(BaseModel):
    """V1 catalog schema."""

    id: UUID
    organization_id: UUID
    name: str
    attributes: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    format: str
    summary: Optional[str]
    description: Optional[str]
    use_count: int
    status: CatalogStatus
    status_reason: Optional[str]
    deleted: bool
    deleted_at: Optional[datetime]
    files: List[V1CheckpointFile]
    created_at: datetime


class V1CheckpointFile(BaseModel):
    """V1 checkpoint file schema."""

    name: str
    path: str
    mtime: datetime
    size: int
    created_at: datetime


class V1ModelForm(BaseModel):
    """V1 model form schema."""

    id: UUID
    form_category: str
    secret_id: Optional[UUID]
    secret_type: str
    vendor: str
    region: str
    storage_name: str
    dist_json: Optional[Dict[str, Any]]
    files: List[V1CheckpointFile]
    created_at: datetime
    deleted: bool
    deleted_at: Optional[datetime]
    hard_deleted: bool
    hard_deleted_at: Optional[datetime]


class V1CheckpointOwnership(BaseModel):
    """V1 checkpoint ownership schema."""

    organization_id: UUID
    project_id: UUID


class V1Checkpoint(BaseModel):
    """V1 checkpoint schema."""

    model_config = ConfigDict(protected_namespaces=())

    id: UUID
    user_id: UUID
    model_category: CheckpointCategory
    job_id: Optional[UUID]
    name: str
    attributes: Optional[Dict[str, Any]]
    iteration: Optional[int]
    tags: Optional[List[str]]
    catalog: Optional[V1Catalog]
    status: CheckpointStatus
    status_reason: Optional[str]
    validation_status: Optional[CheckpointValidationStatus]
    validation_status_reason: Optional[str]
    created_at: datetime
    deleted: bool
    deleted_at: Optional[datetime]
    hard_deleted: bool
    hard_deleted_at: Optional[datetime]
    forms: Optional[List[V1ModelForm]] = None
    ownerships: Optional[List[V1CheckpointOwnership]] = None
