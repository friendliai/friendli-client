# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Quantizer Maps."""

from __future__ import annotations

from typing import Any, Dict, Type

from friendli.enums import QuantMode
from friendli.errors import NotSupportedQuantModeError
from friendli.modules.converter.base import OneOfConverter
from friendli.modules.converter.utils import get_model_arch
from friendli.modules.quantizer.awq.base import AWQHook, AWQQuantizer
from friendli.modules.quantizer.awq.models.gpt_neox import AWQGPTNeoXHook
from friendli.modules.quantizer.awq.models.gptj import AWQGPTJHook
from friendli.modules.quantizer.awq.models.llama import AWQLlamaHook
from friendli.modules.quantizer.awq.models.mpt import AWQMPTHook
from friendli.modules.quantizer.base import CommonQuantizer, FP8QuantHook, FP8Quantizer
from friendli.modules.quantizer.models.llama import LlamaHook
from friendli.modules.quantizer.models.mixtral import MixtralHook
from friendli.modules.quantizer.models.mpt import MPTHook
from friendli.modules.quantizer.schema.config import OneOfQuantConfig
from friendli.modules.quantizer.smoothquant.base import (
    SmoothQuantHook,
    SmoothQuantQuantizer,
)
from friendli.modules.quantizer.smoothquant.models.bloom import SmoothQuantBloomHook
from friendli.modules.quantizer.smoothquant.models.codegen import SmoothQuantCodeGenHook
from friendli.modules.quantizer.smoothquant.models.falcon import SmoothQuantFalconHook
from friendli.modules.quantizer.smoothquant.models.gpt2 import SmoothQuantGPT2Hook
from friendli.modules.quantizer.smoothquant.models.gpt_neox import (
    SmoothQuantGPTNeoXHook,
)
from friendli.modules.quantizer.smoothquant.models.gptj import SmoothQuantGPTJHook
from friendli.modules.quantizer.smoothquant.models.llama import SmoothQuantLlamaHook
from friendli.modules.quantizer.smoothquant.models.mpt import SmoothQuantMPTHook
from friendli.modules.quantizer.smoothquant.models.opt import SmoothQuantOPTHook

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

model_arch_common_hook_map: Dict[str, type[FP8QuantHook]] = {
    "LlamaForCausalLM": LlamaHook,
    "MistralForCausalLM": LlamaHook,
    "MixtralForCausalLM": MixtralHook,
    "MPTForCausalLM": MPTHook,
}


def get_quanthook_map(quant_mode: QuantMode) -> Dict[str, Any]:
    """Get quantizer map."""
    if quant_mode == QuantMode.SMOOTH_QUANT:
        return model_arch_smoothquant_hook_map
    if quant_mode == QuantMode.AWQ:
        return model_arch_awq_hook_map
    if quant_mode == QuantMode.FP8:
        return model_arch_common_hook_map
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
    if quant_mode == QuantMode.FP8:
        return FP8Quantizer
    raise NotSupportedQuantModeError(
        invalid_option=quant_mode,
        valid_options=[e.value for e in QuantMode],
    )


def get_quantized_converter(
    quant_config: OneOfQuantConfig,
    converter: OneOfConverter,
) -> CommonQuantizer:
    """Get quantizer for specific model architecture with quant mode and args."""
    model_arch = get_model_arch(converter.config)
    quant_mode = quant_config.mode
    quantizer = get_quantizer_class(quant_mode)
    quanthook_map = get_quanthook_map(quant_mode)
    quanthook = quanthook_map[model_arch](quant_config, converter)
    return quantizer(quanthook, quant_config, converter)
