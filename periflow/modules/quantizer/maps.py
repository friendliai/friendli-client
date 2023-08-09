# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Maps."""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig  # type: ignore[import]

from periflow.modules.quantizer.base import SmoothQuantHook, SmoothQuantQuantizer
from periflow.modules.quantizer.models.bloom import SmoothQuantBloomHook
from periflow.modules.quantizer.models.codegen import SmoothQuantCodeGenHook
from periflow.modules.quantizer.models.falcon import SmoothQuantFalconHook
from periflow.modules.quantizer.models.gpt2 import SmoothQuantGPT2Hook
from periflow.modules.quantizer.models.gpt_neox import SmoothQuantGPTNeoXHook
from periflow.modules.quantizer.models.gptj import SmoothQuantGPTJHook
from periflow.modules.quantizer.models.llama import SmoothQuantLlamaHook
from periflow.modules.quantizer.models.mpt import SmoothQuantMPTHook
from periflow.modules.quantizer.models.opt import SmoothQuantOPTHook

model_arch_smoothquant_hook_map: Dict[str, type[SmoothQuantHook]] = {
    "OPTForCausalLM": SmoothQuantOPTHook,
    "MPTForCausalLM": SmoothQuantMPTHook,
    "BloomForCausalLM": SmoothQuantBloomHook,
    "CodeGenForCausalLM": SmoothQuantCodeGenHook,
    "GPTNeoXForCausalLM": SmoothQuantGPTNeoXHook,
    "GPTJForCausalLM": SmoothQuantGPTJHook,
    "GPT2LMHeadModel": SmoothQuantGPT2Hook,
    "FalconForCausalLM": SmoothQuantFalconHook,
    "LlamaForCausalLM": SmoothQuantLlamaHook,
}


def get_smoothquant_quantizer(
    model_arch: str, model_config: PretrainedConfig, smoothquant_config: Dict[str, Any]
) -> SmoothQuantQuantizer:
    """Get SmoothQuantQuantizer for specific model architecture."""
    return SmoothQuantQuantizer(
        model_arch_smoothquant_hook_map[model_arch](model_config), smoothquant_config
    )


def check_support_smoothquant(model_arch: str) -> bool:
    """Check if model architecture supports SmoothQuant."""
    return model_arch in model_arch_smoothquant_hook_map
