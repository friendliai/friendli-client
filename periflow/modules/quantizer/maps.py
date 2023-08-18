# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Maps."""

from __future__ import annotations

from typing import Any, Dict

from periflow.modules.quantizer.base import SmoothQuantHook, SmoothQuantQuantizer
from periflow.modules.quantizer.models.mpt import SmoothQuantMPTHook
from periflow.modules.quantizer.models.opt import SmoothQuantOPTHook

model_arch_smoothquant_hook_map: Dict[str, SmoothQuantHook] = {
    "OPTForCausalLM": SmoothQuantOPTHook(),
    "MPTForCausalLM": SmoothQuantMPTHook(),
}


def get_smoothquant_quantizer(
    model_arch: str, config: Dict[str, Any]
) -> SmoothQuantQuantizer:
    """Get SmoothQuantQuantizer for specific model architecture."""
    return SmoothQuantQuantizer(model_arch_smoothquant_hook_map[model_arch], config)


def check_support_smoothquant(model_arch: str) -> bool:
    """Check if model architecture supports SmoothQuant."""
    return model_arch in model_arch_smoothquant_hook_map
