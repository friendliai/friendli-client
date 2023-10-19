# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Defining PeriFlow Checkpoint Converter maps."""

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
    OPTForCausalLM,
    PreTrainedModel,
    T5ForConditionalGeneration,
)

from periflow.errors import NotSupportedCheckpointError
from periflow.modules.converter.base import OneOfConverter
from periflow.modules.converter.models.blenderbot import BlenderbotConverter
from periflow.modules.converter.models.bloom import BloomForCausalLMConverter
from periflow.modules.converter.models.codegen import CodegenForCausalLMConverter
from periflow.modules.converter.models.falcon import FalconForCausalLMConverter
from periflow.modules.converter.models.gpt2 import GPT2LMHeadModelConverter
from periflow.modules.converter.models.gpt_neox import GPTNeoXForCausalLMConverter
from periflow.modules.converter.models.gptj import GPTJForCausalLMConverter
from periflow.modules.converter.models.llama import LlamaForCausalLMConverter
from periflow.modules.converter.models.mpt import MPTForCausalLMConverter
from periflow.modules.converter.models.opt import OPTForCausalLMConverter
from periflow.modules.converter.models.t5 import T5Converter

model_arch_converter_map: Dict[
    str, Tuple[Union[AutoModelForCausalLM, PreTrainedModel], Type[OneOfConverter]]
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
    "MPTForCausalLM": (AutoModelForCausalLM, MPTForCausalLMConverter),
    "OPTForCausalLM": (OPTForCausalLM, OPTForCausalLMConverter),
    "T5ForConditionalGeneration": (T5ForConditionalGeneration, T5Converter),
}


def get_hf_converter_factory(
    model_arch: str,
) -> Tuple[Union[AutoModelForCausalLM, PreTrainedModel], Type[OneOfConverter]]:
    """Return the converter factory for the given model architecture.

    Args:
        model_arch (str): Model architecture name.

    Returns:
        Tuple[Union[AutoModelForCausalLM, PreTrainedModel], Type[OneOfConverter]]: Tuple of
            model class and converter class.

    Raises:
        NotSupportedCheckpointError: Raised when the given model architecture is not supported.

    """
    if model_arch not in model_arch_converter_map:
        raise NotSupportedCheckpointError(
            invalid_option=f"Model architecture='{model_arch}'",
            valid_options=list(model_arch_converter_map.keys()),
        )

    return model_arch_converter_map[model_arch]
