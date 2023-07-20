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

from periflow.converter.base import AbstractConverter
from periflow.converter.models.blenderbot import BlenderbotConverter
from periflow.converter.models.bloom import BloomForCausalLMConverter
from periflow.converter.models.codegen import CodegenForCausalLMConverter
from periflow.converter.models.falcon import FalconForCausalLMConverter
from periflow.converter.models.gpt2 import GPT2LMHeadModelConverter
from periflow.converter.models.gpt_neox import GPTNeoXForCausalLMConverter
from periflow.converter.models.gptj import GPTJForCausalLMConverter
from periflow.converter.models.llama import LlamaForCausalLMConverter
from periflow.converter.models.mpt import MPTForCausalLMConverter
from periflow.converter.models.opt import OPTForCausalLMConverter
from periflow.converter.models.t5 import T5Converter

model_arch_converter_map: Dict[
    str, Tuple[Union[AutoModelForCausalLM, PreTrainedModel], Type[AbstractConverter]]
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
