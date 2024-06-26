# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Model Quantizer Enums."""


from __future__ import annotations

from enum import Enum


class QuantMode(str, Enum):
    """Supported quantization modes."""

    INT8 = "int8"
    DUMMY = "dummy"


class QuantDatasetFormat(str, Enum):
    """Supported file format for calibration datasets for quantization."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    TXT = "txt"


class Int8QuantType(str, Enum):
    """Int8Quant modes."""

    DYNAMIC = "dynamic"


class ModelDataType(str, Enum):
    """Model dtype enums."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8_E4M3 = "fp8_e4m3"
    INT8 = "int8"
    INT4 = "int4"
