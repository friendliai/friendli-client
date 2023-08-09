# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Formatter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

ModuleName = str


@dataclass
class Int8QuantScale:
    """Dataclass for int8 quantization result per layer."""

    module_name: str
    in_scale: float
    weight_scale: float
    out_scale: float
    int8_weight: torch.Tensor  # [OutDim, InDim]


Int8QuantScaleInputTuple = Tuple[torch.Tensor, ModuleName, Optional[int], Optional[int]]
# tuple of (weight in the shape of [OutDim, InDim],
# layer's module name, out_offset_start, out_offset_end)


@dataclass
class Int8QuantScaleInput:
    """Dataclass for int8 quantization input per layer."""

    layer_index: int
    q: Int8QuantScaleInputTuple
    k: Int8QuantScaleInputTuple
    v: Int8QuantScaleInputTuple
    attn_fc: Int8QuantScaleInputTuple
    ff1: Int8QuantScaleInputTuple
    ff2: Int8QuantScaleInputTuple


@dataclass
class Int8QuantScaleResult:
    """Dataclass for int8 quantization result per a transformer layer."""

    layer_index: int
    q: Int8QuantScale
    k: Int8QuantScale
    v: Int8QuantScale
    attn_fc: Int8QuantScale
    ff1: Int8QuantScale
    ff2: Int8QuantScale
