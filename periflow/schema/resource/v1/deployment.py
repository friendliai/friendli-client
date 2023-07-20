# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Deployment Schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from periflow.enums import DeploymentReplicaStatus, DeploymentStatus, DeploymentType


class V1VirtualMachine(BaseModel):
    """V1 virtual machine schema."""

    name: Optional[str]
    gpu_type: Optional[str]
    gpu_memory: float
    allocated_gpus: int
    vcpu: int
    cpu_memory: float


class InferenceServer(BaseModel):
    """V1 inference server schema."""

    name: str
    repo: str
    tag: str


class V1DeploymentConfig(BaseModel):
    """V1 deployment config schema."""

    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., max_length=127)
    description: Optional[str] = Field(None, max_length=127)
    model_id: UUID
    deployment_type: DeploymentType
    vm: V1VirtualMachine
    project_id: UUID
    inference_server_type: InferenceServer
    orca_config: Optional[Dict[str, Any]]
    triton_config: Optional[Dict[str, Any]]
    proxy_rate_limit: int = Field(0, ge=0)
    download_ckpt: bool
    total_gpus: int = Field(1, ge=0)
    ckpt_config: Dict[str, Any]
    cloud: str
    region: str
    cluster_name: Optional[str] = Field(None, max_length=127)
    scaler_config: Dict[str, Any]
    infrequest_perm_check: bool
    infrequest_log: bool


class V1Deployment(BaseModel):
    """V1 deployment schema."""

    deployment_id: str
    version: int
    update_msg: str
    description: Optional[str]
    namespace: str
    user_id: UUID
    project_id: UUID
    config: V1DeploymentConfig
    type: DeploymentType
    ready_replicas: int
    replica_status: List[DeploymentReplicaStatus]
    status: DeploymentStatus
    vms: List[V1VirtualMachine]
    start: datetime
    end: Optional[datetime]
    endpoint: str  # TODO: Use HttpUrl
    error_msg: str
    updated_time: datetime
