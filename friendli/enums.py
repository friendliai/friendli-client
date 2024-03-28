# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli client enums."""


from __future__ import annotations

from enum import Enum


class ModelDataType(str, Enum):
    """Model dtype enums."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8_E4M3 = "fp8_e4m3"
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
    FP8 = "fp8"


class QuantDatasetFormat(str, Enum):
    """Supported file format for calibration datasets for quantization."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    TXT = "txt"


class ResponseFormat(str, Enum):
    """Response formats of text-to-image."""

    URL = "url"
    PNG = "png"
    JPEG = "jpeg"
    RAW = "raw"
