# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Credential Schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Union
from uuid import UUID

from pydantic import BaseModel, EmailStr, HttpUrl, SecretStr
from typing_extensions import TypeAlias


class V1S3Credential(BaseModel):
    """AWS S3 credential schema."""

    aws_access_key_id: SecretStr
    aws_secret_access_key: SecretStr
    aws_default_region: str


class V1AzureBlobCredential(BaseModel):
    """Azure Blob Storage credential schema."""

    storage_account_name: str
    storage_account_key: SecretStr


class V1GCSCredential(BaseModel):
    """Google Cloud Storage credential schema."""

    project_id: str
    private_key_id: SecretStr
    private_key: SecretStr
    client_email: EmailStr
    client_id: str
    auth_uri: HttpUrl
    token_uri: HttpUrl
    auth_provider_x509_cert_url: HttpUrl
    client_x509_cert_url: HttpUrl


CredentialValue: TypeAlias = Union[
    V1S3Credential,
    V1AzureBlobCredential,
    V1GCSCredential,
]


class V1Credential(BaseModel):
    """V1 credential schema."""

    id: UUID
    name: str
    type: str
    type_version: int
    value: Optional[CredentialValue]
    owner_type: str
    owner_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
