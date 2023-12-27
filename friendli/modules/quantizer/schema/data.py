# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Checkpoint Quantizer Data Schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

ModuleName = str


@dataclass
class CommonQuantResult:
    """Dataclass for quantization result per layer."""

    module_name: str
    q_bit: int
    q_group_size: int
    zero_point: torch.Tensor


@dataclass
class WeightOnlyQuantResult(CommonQuantResult):
    """Dataclass for weight-only quantization result per layer."""

    weight_scale: torch.Tensor
    q_weight: torch.Tensor


@dataclass
class WeightActQuantResult(WeightOnlyQuantResult):
    """Dataclass for weight-activation quantization result per layer."""

    in_scale: torch.Tensor
    out_scale: torch.Tensor


@dataclass
class QuantInput:
    """Dataclass for int8 quantization input of each layer in transformer block."""

    weight: torch.Tensor  # [OutDim, InDim]
    name: ModuleName
    start_offset: Optional[int]  # start offset of the weight tensor along the out_dim
    end_offset: Optional[int]  # end offset of the weight tensor along the out_dim
    sort_fn: Optional[
        Callable[[torch.Tensor], torch.Tensor]
    ] = None  # sort function for max_output_stats


@dataclass
class TFQuantInputs:  # pylint: disable=too-many-instance-attributes
    """Dataclass for int8 quantization input per transformer block."""

    layer_index: int
    block: torch.nn.Module
    q: QuantInput
    k: QuantInput
    v: QuantInput
    attn_fc: QuantInput
    ff1: QuantInput
    ff2: QuantInput


@dataclass
class TFQuantResults:  # pylint: disable=too-many-instance-attributes
    """Dataclass for int8 quantization result per a transformer block."""

    layer_prefix_with_index: str
    block: torch.nn.Module
    q: CommonQuantResult
    k: CommonQuantResult
    v: CommonQuantResult
    attn_fc: CommonQuantResult
    ff1: CommonQuantResult
    ff2: CommonQuantResult
