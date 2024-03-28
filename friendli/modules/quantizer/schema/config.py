# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Model Quantizer Config Schema."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from friendli.enums import ModelDataType, QuantDatasetFormat, QuantMode


class CalibrationDatasetConfig(BaseModel):
    """Calibration dataset config."""

    path_or_name: str = "lambada"
    format: QuantDatasetFormat = QuantDatasetFormat.JSON
    split: str = "validation"
    lookup_column_name: str = "text"
    num_samples: int = 128
    max_length: int = 512


class AbstractQuantConfig(BaseModel):
    """Abstract quantization config."""

    mode: QuantMode
    device: str = "cuda:0"
    offload: bool = True
    seed: int = 42
    percentile: float = 100.0
    quant_dtype: ModelDataType = ModelDataType.INT8
    calibration_dataset: CalibrationDatasetConfig = Field(
        default_factory=CalibrationDatasetConfig
    )


class FP8QuantConfig(AbstractQuantConfig):
    """FP8 quantization config.

    The data type of parameters are converted to the one specified at `quant_dtype`
    by using calibration dataset. The quantization scale for weight and activation is
    added to converted checkpoint.

    """

    mode: Literal[QuantMode.FP8] = QuantMode.FP8


class SmoothQuantArgs(BaseModel):
    """SmoothQuant args."""

    migration_strength: float = 0.5
    attn_fc_smoothing: bool = False
    ff2_smoothing: bool = False


class SmoothQuantConfig(AbstractQuantConfig):
    """SmoothQuant config."""

    mode: Literal[QuantMode.SMOOTH_QUANT] = QuantMode.SMOOTH_QUANT
    smoothquant_args: SmoothQuantArgs = Field(default_factory=SmoothQuantArgs)


class AWQArgs(BaseModel):
    """AWQ args."""

    quant_dtype: ModelDataType = ModelDataType.INT4
    quant_bit: int = 4
    quant_group_size: int = 64


class AWQConfig(AbstractQuantConfig):
    """AWQ config."""

    mode: Literal[QuantMode.AWQ] = QuantMode.AWQ
    awq_args: AWQArgs = Field(default_factory=AWQArgs)


OneOfQuantConfig = Annotated[
    Union[SmoothQuantConfig, AWQConfig, FP8QuantConfig], Field(discriminator="mode")
]


class QuantConfig(BaseModel):
    """Quantization config."""

    config: OneOfQuantConfig
