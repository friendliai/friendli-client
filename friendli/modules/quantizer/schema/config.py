# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Checkpoint Quantizer Config Schema."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from friendli.enums import QuantDatasetFormat, QuantMode


class CalibrationDatasetConfig(BaseModel):
    """Calibration dataset config."""

    path_or_name: str = "lambada"
    format: QuantDatasetFormat = QuantDatasetFormat.JSON
    split: str = "validation"
    lookup_column_name: str = "text"
    num_samples: int = 128
    max_length: int = 512


class CommonQuantConfig(BaseModel):
    """Common quantization config."""

    mode: QuantMode
    device: str = "cuda:0"
    offload: bool = True
    seed: int = 42
    percentile: float = 99.9
    calibration_dataset: CalibrationDatasetConfig = Field(
        default_factory=CalibrationDatasetConfig
    )


class SmoothQuantArgs(BaseModel):
    """SmoothQuant args."""

    migration_strength: float = 0.5
    attn_fc_smoothing: bool = False
    ff2_smoothing: bool = False


class SmoothQuantConfig(CommonQuantConfig):
    """SmoothQuant config."""

    mode: Literal[QuantMode.SMOOTH_QUANT] = QuantMode.SMOOTH_QUANT
    smoothquant_args: SmoothQuantArgs = Field(default_factory=SmoothQuantArgs)


class AWQArgs(BaseModel):
    """AWQ args."""

    quant_bit: int = 4
    quant_group_size: int = 64


class AWQConfig(CommonQuantConfig):
    """AWQ config."""

    mode: Literal[QuantMode.AWQ] = QuantMode.AWQ
    awq_args: AWQArgs = Field(default_factory=AWQArgs)


OneOfQuantConfig = Annotated[
    Union[SmoothQuantConfig, AWQConfig], Field(discriminator="mode")
]


class QuantConfig(BaseModel):
    """Quantization config."""

    config: OneOfQuantConfig
