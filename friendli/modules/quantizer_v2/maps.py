# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Quantizer V2 Maps."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type, cast

import transformers  # type: ignore
from transformers import (  # type: ignore
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

from friendli.errors import NotSupportedQuantModeError, QuantizationError
from friendli.modules.quantizer_v2.base import AbstractQuantizerV2
from friendli.modules.quantizer_v2.enums import Int8QuantType, QuantMode
from friendli.modules.quantizer_v2.int8.base import Int8DynamicQuantizer, Int8QuantHook
from friendli.modules.quantizer_v2.models.llama import LlamaInt8QuantHook
from friendli.modules.quantizer_v2.models.phi3 import Phi3Int8QuantHook
from friendli.modules.quantizer_v2.schema.config import (
    Int8QuantConfig,
    OneOfQuantConfig,
)

model_arch_int8_hook_map: Dict[PreTrainedModel, type[Int8QuantHook]] = {
    LlamaForCausalLM: LlamaInt8QuantHook,
    MistralForCausalLM: LlamaInt8QuantHook,
    Phi3ForCausalLM: Phi3Int8QuantHook,
}


def get_quanthook_map(quant_mode: QuantMode) -> Dict[Type[PreTrainedModel], Any]:
    """Get quantizer map."""
    if quant_mode == QuantMode.INT8:
        return model_arch_int8_hook_map
    raise NotSupportedQuantModeError(
        invalid_option=quant_mode,
        valid_options=[e.value for e in QuantMode],
    )


def get_model_class(config: PretrainedConfig) -> PreTrainedModel:
    """Get HuggingFace model architecture from config."""
    model_arch_list = cast(List[str], cast(PretrainedConfig, config).architectures)
    if len(model_arch_list) == 0:
        raise QuantizationError("Model architecture not found in config.")
    model_arch = model_arch_list[0]
    try:
        cls_type = getattr(transformers, model_arch, None)
    except AttributeError as exc:
        raise QuantizationError(str(exc)) from exc
    return cls_type


def get_quantizer_class(quant_config: OneOfQuantConfig) -> Type[AbstractQuantizerV2]:
    """Get quantizer class."""
    quant_mode = quant_config.mode
    if quant_mode == QuantMode.INT8:
        if (
            cast(Int8QuantConfig, quant_config).int8_args.quant_type
            == Int8QuantType.DYNAMIC
        ):
            return Int8DynamicQuantizer
        raise QuantizationError(
            "Only Dynamic quantization is supported for int8 quantization."
        )
    raise NotSupportedQuantModeError(
        invalid_option=quant_mode,
        valid_options=[e.value for e in QuantMode],
    )


def get_hf_quantizer_factory(
    model_config: PretrainedConfig,
    quant_config: OneOfQuantConfig,
) -> Tuple[PreTrainedModel, AbstractQuantizerV2]:
    """Get quantizer for specific model architecture with quant mode and args."""
    hf_model_cls = get_model_class(model_config)
    quantizer = get_quantizer_class(quant_config)
    quanthook_map = get_quanthook_map(quant_config.mode)
    quanthook = quanthook_map[hf_model_cls](quant_config, model_config)
    return hf_model_cls, quantizer(quanthook, quant_config)
