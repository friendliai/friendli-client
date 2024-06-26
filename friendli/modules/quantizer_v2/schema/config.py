# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Model Quantizer Config Schema."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from friendli.modules.quantizer_v2.enums import (
    Int8QuantType,
    ModelDataType,
    QuantDatasetFormat,
    QuantMode,
)


class CalibrationDatasetConfig(BaseModel):
    """Calibration dataset config."""

    path_or_name: str = "cnn_dailymail:3.0.0"
    format: QuantDatasetFormat = QuantDatasetFormat.JSON
    split: str = "validation"
    lookup_column_name: str = "article"
    num_samples: int = 512
    max_length: int = 512
    batch_size: int = 1


class AbstractQuantConfig(BaseModel):
    """Abstract quantization config."""

    mode: QuantMode
    device: str = "cuda:0"
    offload: bool = True
    seed: int = 42
    percentile: float = 100.0
    quant_dtype: ModelDataType = ModelDataType.INT8
    quant_scale_dtype: Optional[ModelDataType] = None
    use_symmetric: bool = True
    quant_group_size: int = -1  # no grouping
    calibration_dataset: CalibrationDatasetConfig = Field(
        default_factory=CalibrationDatasetConfig
    )


class Int8QuantArtgs(BaseModel):
    """Int8Quant args."""

    migration_strength: float = 0.5
    quant_type: Int8QuantType = Int8QuantType.DYNAMIC


class Int8QuantConfig(AbstractQuantConfig):
    """Int8Quant config."""

    mode: Literal[QuantMode.INT8] = QuantMode.INT8
    int8_args: Int8QuantArtgs = Field(default_factory=Int8QuantArtgs)


class DummyQuantConfig(AbstractQuantConfig):
    """Dummy quant config."""

    mode: Literal[QuantMode.DUMMY] = QuantMode.DUMMY


OneOfQuantConfig = Annotated[
    Union[Int8QuantConfig, DummyQuantConfig], Field(discriminator="mode")
]


class QuantConfig(BaseModel):
    """Quantization config."""

    config: OneOfQuantConfig
