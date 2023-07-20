# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Defines mapping."""

from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from periflow.enums import CredType
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
