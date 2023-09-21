# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Formatter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from periflow.enums import QuantDatasetFormat, QuantMode

ModuleName = str


@dataclass
class Int8QuantResult:
    """Dataclass for int8 quantization result per layer."""

    module_name: str
    in_scale: float
    weight_scale: float
    out_scale: float
    int8_weight: torch.Tensor  # [OutDim, InDim]


@dataclass
class Int8QuantInput:
    """Dataclass for int8 quantization input of each layer in transformer block."""

    weight: torch.Tensor  # [OutDim, InDim]
    name: ModuleName
    start_offset: Optional[int]  # start offset of the weight tensor along the out_dim
    end_offset: Optional[int]  # end offset of the weight tensor along the out_dim
    sort_fn: Optional[
        Callable[[torch.Tensor], torch.Tensor]
    ] = None  # sort function for max_output_stats


@dataclass
class TFInt8QuantInputs:
    """Dataclass for int8 quantization input per transformer block."""

    layer_index: int
    q: Int8QuantInput
    k: Int8QuantInput
    v: Int8QuantInput
    attn_fc: Int8QuantInput
    ff1: Int8QuantInput
    ff2: Int8QuantInput


@dataclass
class TFInt8QuantResults:
    """Dataclass for int8 quantization result per a transformer block."""

    layer_index: int
    q: Int8QuantResult
    k: Int8QuantResult
    v: Int8QuantResult
    attn_fc: Int8QuantResult
    ff1: Int8QuantResult
    ff2: Int8QuantResult


class CalibrationDatasetConfig(BaseModel):
    """Calibration dataset config."""

    path_or_name: str = "lambada"
    format: QuantDatasetFormat = QuantDatasetFormat.JSON
    split: str = "validation"
    lookup_column_name: str = "text"
    num_samples: int = 512
    max_length: int = 512


class CommonQuantConfig(BaseModel):
    """Common quantization config."""

    mode: QuantMode
    device: str = "cuda:0"
    seed: int = 42
    calibration_dataset: CalibrationDatasetConfig = Field(
        default_factory=CalibrationDatasetConfig
    )


class SmoothQuantArgs(BaseModel):
    """SmoothQuant args."""

    migration_strength: float = 0.5


class SmoothQuantConfig(CommonQuantConfig):
    """SmoothQuant config."""

    mode: Literal[QuantMode.SMOOTH_QUANT] = QuantMode.SMOOTH_QUANT
    smoothquant_args: SmoothQuantArgs = Field(default_factory=SmoothQuantArgs)


# Added for utilizing discriminated union.
class MockConfig(CommonQuantConfig):
    """HACK: Will be removed after adding another quantization scheme."""

    mode: Literal[QuantMode.NONE] = QuantMode.NONE


OneOfQuantConfig = Annotated[
    Union[SmoothQuantConfig, MockConfig], Field(discriminator="mode")
]


class QuantConfig(BaseModel):
    """Quantization config."""

    config: OneOfQuantConfig
