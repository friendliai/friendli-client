# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Defines mapping."""

from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from periflow.enums import CloudType, CredType, GpuType, VMType
from periflow.schema.resource.v1.credential import (
    V1AzureBlobCredential,
    V1GCSCredential,
    V1S3Credential,
)

cred_type_map: Dict[CredType, str] = {
    CredType.S3: "aws",
    CredType.BLOB: "azure.blob",
    CredType.GCS: "gcp",
}


cred_type_map_inv: Dict[str, CredType] = {
    "aws": CredType.S3,
    "azure.blob": CredType.BLOB,
    "gcp": CredType.GCS,
}


cred_schema_map: Dict[CredType, Type[BaseModel]] = {
    CredType.S3: V1S3Credential,
    CredType.BLOB: V1AzureBlobCredential,
    CredType.GCS: V1GCSCredential,
}


cloud_vm_map: Dict[CloudType, list[VMType]] = {
    CloudType.AWS: [VMType.G5_XLARGE],
    CloudType.AZURE: [],
    CloudType.GCP: [
        VMType.A2_HIGHGPU_1G,
        VMType.A2_ULTRAGPU_1G,
        VMType.A2_ULTRAGPU_2G,
        VMType.A2_ULTRAGPU_4G,
        VMType.A2_ULTRAGPU_8G,
    ],
}


cloud_gpu_map: Dict[CloudType, list[GpuType]] = {
    CloudType.AWS: [GpuType.A10G],
    CloudType.AZURE: [GpuType.A100_80G],
    CloudType.GCP: [GpuType.A100],
}


gpu_num_map: Dict[GpuType, list[int]] = {
    GpuType.A10G: [1],
    GpuType.A100_80G: [1, 2, 4, 8],
    GpuType.A100: [1],
}


vm_num_gpu_map: Dict[VMType, int] = {
    VMType.G5_XLARGE: 1,
    VMType.A2_HIGHGPU_1G: 1,
    VMType.A2_ULTRAGPU_1G: 1,
    VMType.A2_ULTRAGPU_2G: 2,
    VMType.A2_ULTRAGPU_4G: 4,
    VMType.A2_ULTRAGPU_8G: 8,
}
