# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Model Quantizer Data Schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from friendli.enums import ModelDataType

ModuleName = str


@dataclass
class CommonQuantResult:
    """Dataclass for quantization result per layer."""

    module_name: str
    quant_dtype: ModelDataType
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

    act_scale: torch.Tensor
    zero_point: torch.Tensor
    q_group_size: int


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
class HFQuantInput:
    """Dataclass for quantization input of each layer in transformer block.

    Attributes:
        parent_module: module contains target layers.
        target_names: list of target module's full name
                    (ex. model.model.layers.0.self_attn.q_proj, )
        local_names: list of target module's name using when access from parent_module
                    (ex. q_proj, k_proj, v_proj )
    """

    parent_module: torch.nn.Module
    target_names: List[ModuleName]
    local_names: str


@dataclass
class HFTFQuantInputs:
    """Dataclass for quantization input per transformer block."""

    layer_index: int
    block: torch.nn.Module
    quant_inputs: List[HFQuantInput]


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
