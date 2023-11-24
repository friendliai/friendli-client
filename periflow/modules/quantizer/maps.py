# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Maps."""

from __future__ import annotations

from typing import Any, Dict, Type

from periflow.enums import QuantMode
from periflow.errors import NotSupportedQuantModeError
from periflow.modules.converter.base import OneOfConverter
from periflow.modules.quantizer.awq.base import AWQHook, AWQQuantizer
from periflow.modules.quantizer.awq.models.gpt_neox import AWQGPTNeoXHook
from periflow.modules.quantizer.awq.models.gptj import AWQGPTJHook
from periflow.modules.quantizer.awq.models.llama import AWQLlamaHook
from periflow.modules.quantizer.awq.models.mpt import AWQMPTHook
from periflow.modules.quantizer.base import CommonQuantizer
from periflow.modules.quantizer.schema.config import OneOfQuantConfig
from periflow.modules.quantizer.smoothquant.base import (
    SmoothQuantHook,
    SmoothQuantQuantizer,
)
from periflow.modules.quantizer.smoothquant.models.bloom import SmoothQuantBloomHook
from periflow.modules.quantizer.smoothquant.models.codegen import SmoothQuantCodeGenHook
from periflow.modules.quantizer.smoothquant.models.falcon import SmoothQuantFalconHook
from periflow.modules.quantizer.smoothquant.models.gpt2 import SmoothQuantGPT2Hook
from periflow.modules.quantizer.smoothquant.models.gpt_neox import (
    SmoothQuantGPTNeoXHook,
)
from periflow.modules.quantizer.smoothquant.models.gptj import SmoothQuantGPTJHook
from periflow.modules.quantizer.smoothquant.models.llama import SmoothQuantLlamaHook
from periflow.modules.quantizer.smoothquant.models.mpt import SmoothQuantMPTHook
from periflow.modules.quantizer.smoothquant.models.opt import SmoothQuantOPTHook

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

model_arch_awq_hook_map: Dict[str, type[AWQHook]] = {
    "GPTJForCausalLM": AWQGPTJHook,
    "GPTNeoXForCausalLM": AWQGPTNeoXHook,
    "LlamaForCausalLM": AWQLlamaHook,
    "MPTForCausalLM": AWQMPTHook,
    "MistralForCausalLM": AWQLlamaHook,
}


def get_quanthook_map(quant_mode: QuantMode) -> Dict[str, Any]:
    """Get quantizer map."""
    if quant_mode == QuantMode.SMOOTH_QUANT:
        return model_arch_smoothquant_hook_map
    if quant_mode == QuantMode.AWQ:
        return model_arch_awq_hook_map
    raise NotSupportedQuantModeError(
        invalid_option=quant_mode,
        valid_options=[e.value for e in QuantMode],
    )


def get_quantizer_class(quant_mode: QuantMode) -> Type[CommonQuantizer]:
    """Get quantizer class."""
    if quant_mode == QuantMode.SMOOTH_QUANT:
        return SmoothQuantQuantizer
    if quant_mode == QuantMode.AWQ:
        return AWQQuantizer
    raise NotSupportedQuantModeError(
        invalid_option=quant_mode,
        valid_options=[e.value for e in QuantMode],
    )


def get_quantized_converter(
    model_arch: str,
    quant_config: OneOfQuantConfig,
    converter: OneOfConverter,
) -> CommonQuantizer:
    """Get quantizer for specific model architecture with quant mode and args."""
    quant_mode = quant_config.mode
    quantizer = get_quantizer_class(quant_mode)
    quanthook_map = get_quanthook_map(quant_mode)
    quanthook = quanthook_map[model_arch](quant_config, converter)
    return quantizer(quanthook, quant_config, converter)
