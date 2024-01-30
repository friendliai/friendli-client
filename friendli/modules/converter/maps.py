# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Defining Friendli Checkpoint Converter maps."""

from __future__ import annotations

from typing import Dict, Tuple, Type, Union

from transformers import (  # type: ignore[import]
    AutoModelForCausalLM,
    BlenderbotForConditionalGeneration,
    BloomForCausalLM,
    CodeGenForCausalLM,
    FalconForCausalLM,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    MixtralForCausalLM,
    MptForCausalLM,
    OPTForCausalLM,
    PreTrainedModel,
    T5ForConditionalGeneration,
)

from friendli.errors import NotSupportedCheckpointError
from friendli.modules.converter.base import OneOfAdapterConverter, OneOfConverter
from friendli.modules.converter.models.blenderbot import BlenderbotConverter
from friendli.modules.converter.models.bloom import BloomForCausalLMConverter
from friendli.modules.converter.models.codegen import CodegenForCausalLMConverter
from friendli.modules.converter.models.falcon import FalconForCausalLMConverter
from friendli.modules.converter.models.gpt2 import GPT2LMHeadModelConverter
from friendli.modules.converter.models.gpt_neox import GPTNeoXForCausalLMConverter
from friendli.modules.converter.models.gptj import (
    GPTJForCausalLMConverter,
    GPTJForCausalLMLoraConverter,
)
from friendli.modules.converter.models.llama import (
    LlamaForCausalLMConverter,
    LlamaForCausalLMLoraConverter,
)
from friendli.modules.converter.models.mistral import MistralForCausalLMConverter
from friendli.modules.converter.models.mixtral import MixtralForCausalLMConverter
from friendli.modules.converter.models.mpt import (
    MPTForCausalLMConverter,
    MptForCausalLMLoraConverter,
)
from friendli.modules.converter.models.opt import OPTForCausalLMConverter
from friendli.modules.converter.models.phi_msft import PhiForCausalLMConverter
from friendli.modules.converter.models.t5 import T5Converter

MODEL_ARCH_CONVERTER_MAP: Dict[
    str, Tuple[Union[PreTrainedModel, PreTrainedModel], Type[OneOfConverter]]
] = {
    "BlenderbotForConditionalGeneration": (
        BlenderbotForConditionalGeneration,
        BlenderbotConverter,
    ),
    "BloomForCausalLM": (BloomForCausalLM, BloomForCausalLMConverter),
    "CodeGenForCausalLM": (CodeGenForCausalLM, CodegenForCausalLMConverter),
    "FalconForCausalLM": (FalconForCausalLM, FalconForCausalLMConverter),
    "GPTNeoXForCausalLM": (GPTNeoXForCausalLM, GPTNeoXForCausalLMConverter),
    "GPT2LMHeadModel": (GPT2LMHeadModel, GPT2LMHeadModelConverter),
    "GPTJForCausalLM": (GPTJForCausalLM, GPTJForCausalLMConverter),
    "LlamaForCausalLM": (LlamaForCausalLM, LlamaForCausalLMConverter),
    "LLaMAForCausalLM": (LlamaForCausalLM, LlamaForCausalLMConverter),
    "MistralForCausalLM": (MistralForCausalLM, MistralForCausalLMConverter),
    "MixtralForCausalLM": (MixtralForCausalLM, MixtralForCausalLMConverter),
    "MPTForCausalLM": (MptForCausalLM, MPTForCausalLMConverter),
    "OPTForCausalLM": (OPTForCausalLM, OPTForCausalLMConverter),
    "T5ForConditionalGeneration": (T5ForConditionalGeneration, T5Converter),
    "PhiForCausalLM": (AutoModelForCausalLM, PhiForCausalLMConverter),
}

MODEL_ARCH_ADAPTER_CONVERTER_MAP: Dict[
    str,
    Type[OneOfAdapterConverter],
] = {
    "GPTJForCausalLM": GPTJForCausalLMLoraConverter,
    "LlamaForCausalLM": LlamaForCausalLMLoraConverter,
    "LLaMAForCausalLM": LlamaForCausalLMLoraConverter,
    "MPTForCausalLM": MptForCausalLMLoraConverter,
}


def get_hf_converter_factory(
    model_arch: str,
) -> Tuple[PreTrainedModel, Type[OneOfConverter]]:
    """Return the converter factory for the given model architecture.

    Args:
        model_arch (str): Model architecture name.

    Returns:
        Tuple[PretrainedModel, Type[OneOfConverter]]: Tuple of
            model class and converter class.

    Raises:
        NotSupportedCheckpointError: Raised when the given model architecture is not supported.

    """
    if model_arch not in MODEL_ARCH_CONVERTER_MAP:
        raise NotSupportedCheckpointError(
            invalid_option=f"Model architecture='{model_arch}'",
            valid_options=list(MODEL_ARCH_CONVERTER_MAP.keys()),
        )

    return MODEL_ARCH_CONVERTER_MAP[model_arch]


def get_adapter_converter_factory(
    model_arch: str,
) -> Type[OneOfAdapterConverter]:
    """Return the converter factory for the given model architecture.

    Args:
        model_arch (str): Model architecture name.

    Returns:
        Type[LoraConverter]: Adapter Converter class.

    Raises:
        NotSupportedCheckpointError: Raised when the given model architecture is not supported.
    """
    try:
        adapter_converter_type = MODEL_ARCH_ADAPTER_CONVERTER_MAP[model_arch]
    except KeyError as exc:
        raise NotSupportedCheckpointError(
            invalid_option=f"adapter for model architecture='{model_arch}'",
            valid_options=list(MODEL_ARCH_ADAPTER_CONVERTER_MAP.keys()),
        ) from exc
    return adapter_converter_type
