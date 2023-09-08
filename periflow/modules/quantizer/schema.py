# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Formatter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

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
