# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Maps."""

from __future__ import annotations

from typing import Dict, Type

from periflow.modules.quantizer.base import SmoothQuantHook
from periflow.modules.quantizer.configurator import (
    QuantConfigurator,
    SmoothQuantConfigurator,
)
from periflow.modules.quantizer.models.bloom import SmoothQuantBloomHook
from periflow.modules.quantizer.models.codegen import SmoothQuantCodeGenHook
from periflow.modules.quantizer.models.falcon import SmoothQuantFalconHook
from periflow.modules.quantizer.models.gpt2 import SmoothQuantGPT2Hook
from periflow.modules.quantizer.models.gpt_neox import SmoothQuantGPTNeoXHook
from periflow.modules.quantizer.models.gptj import SmoothQuantGPTJHook
from periflow.modules.quantizer.models.llama import SmoothQuantLlamaHook
from periflow.modules.quantizer.models.mpt import SmoothQuantMPTHook
from periflow.modules.quantizer.models.opt import SmoothQuantOPTHook

quant_configurator_map: Dict[str, Type[QuantConfigurator]] = {
    "smoothquant": SmoothQuantConfigurator,
}
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
