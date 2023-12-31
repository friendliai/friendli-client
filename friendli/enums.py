# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli client enums."""


from __future__ import annotations

from enum import Enum


class GroupRole(str, Enum):
    """Organization-level roles."""

    OWNER = "owner"
    MEMBER = "member"


class ProjectRole(str, Enum):
    """Project-level roles."""

    ADMIN = "admin"
    MAINTAIN = "maintain"
    DEVELOP = "develop"
    GUEST = "guest"


class ServiceTier(str, Enum):
    """Organization service tier."""

    BASIC = "basic"
    ENT = "enterprise"


class CloudType(str, Enum):
    """Cloud types."""

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class StorageType(str, Enum):
    """Cloud storage types."""

    S3 = "s3"
    BLOB = "azure-blob"
    GCS = "gcs"
    FAI = "fai"


class GpuType(str, Enum):
    """GPU types for deployment."""

    A10G = "a10g"
    A100 = "a100"
    A100_80G = "a100-80g"


class VMType(str, Enum):
    """VM types for deployment."""

    G5_XLARGE = "g5.xlarge"
    A2_HIGHGPU_1G = "a2-highgpu-1g"
    A2_ULTRAGPU_1G = "a2-ultragpu-1g"
    A2_ULTRAGPU_2G = "a2-ultragpu-2g"
    A2_ULTRAGPU_4G = "a2-ultragpu-4g"
    A2_ULTRAGPU_8G = "a2-ultragpu-8g"


class DeploymentType(str, Enum):
    """Deployment phase types."""

    DEV = "dev"
    PROD = "prod"


class DeploymentSecurityLevel(str, Enum):
    """Deployment access levels."""

    PUBLIC = "public"
    PROTECTED = "protected"


class CheckpointCategory(str, Enum):
    """Checkpoint categories."""

    USER_PROVIDED = "USER"
    JOB_GENERATED = "JOB"
    COPIED = "COPY"
    REFERENCED = "REF"


class CheckpointStatus(str, Enum):
    """Checkpoint statuses."""

    CREATED = "Created"  # Model is created
    COPYING = "Copying"  # Copying the model from catalog
    ACTIVE = "Active"  # Available to use
    FAILED = "Failed"  # Cannot use the checkpoint


class CheckpointValidationStatus(str, Enum):
    """Checkpoint validation statuses."""

    VALIDATING = "Validating"  # Validating the checkpoint format
    VALID = "Valid"  # Checkpoint format is valid
    INVALID = "Invalid"  # Checkpoint is invalid
    FAILED = "Failed"  # Validation failed for some reason


class CatalogImportMethod(str, Enum):
    """Catalog import methods."""

    COPY = "copy"
    REF = "ref"


class CatalogStatus(str, Enum):
    """Catalog statuses."""

    CREATED = "Created"  # Catalog is created
    PUBLISHING = "Publishing"  # Publishing the catalog
    ACTIVE = "Active"  # Available to use the catalog
    FAILED = "Failed"  # Failed to publish catalog


class CredType(str, Enum):
    """Credential types."""

    S3 = "s3"
    BLOB = "azure-blob"
    GCS = "gcs"


class DeploymentReplicaStatus(str, Enum):
    """Deployment replica status types."""

    INITIALIZING = "Initializing"
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCESS = "Success"
    FAILED = "Failed"
    UNKNOWN = "Unknown"
    STOPPING = "Stopping"
    TERMINATED = "Terminated"


class DeploymentStatus(str, Enum):
    """Deployment status types."""

    INITIALIZING = "Initializing"
    HEALTHY = "Healthy"
    UNHEALTHY = "Unhealthy"
    FAILED = "Failed"
    STOPPING = "Stopping"
    TERMINATED = "Terminated"


class CheckpointDataType(str, Enum):
    """Checkpoint dtype enums."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"
    INT4 = "int4"


class CheckpointFileType(str, Enum):
    """Checkpoint file types."""

    HDF5 = "hdf5"
    SAFETENSORS = "safetensors"

    def __str__(self):
        """Convert to a human-readable string."""
        return self.value


class FileSizeType(str, Enum):
    """File size type."""

    LARGE = "LARGE"
    SMALL = "SMALL"


class QuantMode(str, Enum):
    """Supported quantization modes."""

    SMOOTH_QUANT = "smoothquant"
    AWQ = "awq"


class QuantDatasetFormat(str, Enum):
    """Supported file format for calibration datasets for quantization."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    TXT = "txt"


class ChatCompletionsModel(str, Enum):
    """Model choices for chat completions."""

    LLAMA_2_13B_CHAT = "llama-2-13b-chat"
    LLAMA_2_70B_CHAT = "llama-2-70b-chat"
    MISTRAL_7B_INSTRUCT_V0_2 = "mistral-7b-instruct-v0-2"


class CompletionsModel(str, Enum):
    """Model choices for completions."""

    LLAMA_2_13B = "llama-2-13b"
    LLAMA_2_70B = "llama-2-70b"
    MISTRAL_7B_V0_1 = "mistral-7b-v0-1"
