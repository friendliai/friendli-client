# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Model Quantizer Data Schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

ModuleName = str


@dataclass
class BaseQuantResult:
    """Dataclass for quantization result per layer."""

    q_group_size: int
    zero_point: Optional[torch.Tensor]
    q_weight: torch.Tensor
    weight_scale: torch.Tensor


@dataclass
class WeightOnlyQuantResult(BaseQuantResult):
    """Dataclass for weight-only quantization result per layer."""


@dataclass
class WeightActQuantResult(BaseQuantResult):
    """Dataclass for weight-activation quantization result per layer."""

    act_scale: torch.Tensor
    q_group_size: int


@dataclass
class QuantInput:
    """Dataclass for quantization input of each layer in transformer block.

    When you want to quantize specific layers at once, the target layers should be
    included in this dataclass. For example, if the quantization scale of the q_proj,
    k_proj, and v_proj layers in the self-attention layer are calculated together,
    the target_names and local_names of these layers should be included in the
    same QuantInput dataclass.

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
class TFQuantInputs:
    """Dataclass for Container of  per transformer block."""

    layer_index: int
    block: torch.nn.Module
    quant_inputs: List[QuantInput]
