# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Dict

import pytest
from peft import PeftConfig
from transformers import (
    AutoConfig,
    BlenderbotConfig,
    BloomConfig,
    CodeGenConfig,
    FalconConfig,
    GPT2Config,
    GPTJConfig,
    GPTNeoXConfig,
    LlamaConfig,
    MistralConfig,
    MixtralConfig,
    MptConfig,
    OPTConfig,
    T5Config,
)
from transformers.models.mpt.configuration_mpt import MptAttentionConfig

from friendli.enums import ModelDataType
from friendli.modules.converter.base import OneOfConverter
from friendli.modules.converter.maps import get_hf_converter_factory
from friendli.modules.converter.models.mixtral import MixtralForCausalLMConverter
from friendli.modules.converter.utils import get_model_arch

from tests.unit_tests.modules.helpers.utils import ModelConfig, get_param_specs

model_name_config_map = {
    "blenderbot": BlenderbotConfig(
        architectures=["BlenderbotForConditionalGeneration"],
        activation_function="gelu",
        tie_word_embeddings=True,
        decoder_attention_heads=32,
        encoder_attention_heads=32,
        decoder_ffn_dim=10240,
        encoder_ffn_dim=10240,
        encoder_layers=1,
        decoder_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "bloom": BloomConfig(
        architectures=["BloomForCausalLM"],
        apply_residual_connection_post_layernorm=False,
        slow_but_exact=False,
        tie_word_embeddings=True,
        layer_norm_epsilon=1e-5,
        n_layer=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "codegen": CodeGenConfig(
        architectures=["CodeGenForCausalLM"],
        activation_function="gelu",
        tie_word_embeddings=False,
        layer_norm_epsilon=1e-5,
        n_layer=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "falcon_7b": FalconConfig(  # falcon-7b
        architectures=["FalconForCausalLM"],
        alibi=False,
        bias=False,
        new_decoder_architecture=False,
        parallel_attn=True,
        num_hidden_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "falcon": FalconConfig(  # falcon-40b
        architectures=["FalconForCausalLM"],
        alibi=False,
        bias=False,
        new_decoder_architecture=True,
        num_hidden_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "gpt_neox": GPTNeoXConfig(  # pythia-1.4b
        architectures=["GPTNeoXForCausalLM"],
        hidden_act="gelu",
        use_parallel_residual=True,
        tie_word_embeddings=False,
        layer_norm_eps=1e-5,
        rotary_emb_base=10000,
        num_hidden_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "gpt": GPT2Config(
        architectures=["GPT2LMHeadModel"],
        activation_function="gelu",
        scale_attn_by_inverse_layer_idx=False,
        tie_word_embeddings=True,
        layer_norm_epsilon=1e-5,
        n_layer=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "gpt_j": GPTJConfig(  # gpt-j-6b
        architectures=["GPTJForCausalLM"],
        tie_word_embeddings=False,
        layer_norm_epsilon=1e-5,
        n_layer=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "llama": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
        num_hidden_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "mpt": MptConfig(
        architectures=["MPTForCausalLM"],
        attn_config=MptAttentionConfig(
            alibi=True,
            alibi_bias_max=8,
            attn_type="multihead_attention",
            prefix_lm=False,
            qk_ln=False,
            softmax_scale=None,
        ),
        expansion_ratio=4,
        no_bias=True,
        logit_scale=None,
        n_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "opt": OPTConfig(
        architectures=["OPTForCausalLM"],
        activation_function="relu",
        do_layer_norm_before=True,
        word_embed_proj_dim=768,
        hidden_size=768,
        _remove_first_dropout=False,
        tie_word_embeddings=True,
        num_hidden_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "t5_v1_1": T5Config(
        architectures=["T5ForConditionalGeneration"],
        is_gated_act=True,
        tie_word_embeddings=False,
        num_hidden_layers=1,
        num_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
        relative_attention_num_buckets=32,  # fixed value for t5
    ),
    "t5": T5Config(
        architectures=["T5ForConditionalGeneration"],
        is_gated_act=False,
        tie_word_embeddings=True,
        layer_norm_epsilon=1e-6,
        num_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
        relative_attention_num_buckets=32,  # fixed value for t5
    ),
    "mistral": MistralConfig(  # same as llama architecture
        architectures=["MistralForCausalLM"],
        hidden_act="silu",
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        num_hidden_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    "mixtral": MixtralConfig(  # same as llama architecture
        architectures=["MixtralForCausalLM"],
        hidden_act="silu",
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        num_hidden_layers=1,
        vocab_size=10000,
        max_position_embeddings=1024,
    ),
    # TODO: add phi_msft
    # TODO: add mpt with grouped querry attention (e.g. replit-code)
}


@pytest.fixture
def converter(model_config: AutoConfig) -> OneOfConverter:
    model_arch = get_model_arch(model_config)
    _, converter_cls = get_hf_converter_factory(model_arch)
    return converter_cls(model_config, None, ModelDataType.FP16)


# TODO: add render_model_config per model
@pytest.fixture
def render_model_config(converter: OneOfConverter) -> ModelConfig:
    return ModelConfig(
        dtype="float16",
        num_decoder_layers=converter.decoder_layer_num,
        hidden_size=converter.decoder_hidden_size,
        num_heads=converter.decoder_num_attention_heads,
        num_kv_heads=converter.decoder_num_kv_attention_heads,
        head_size=converter.decoder_head_size,
        num_encoder_layers=converter.decoder_layer_num,  # same as decoder for test
        ff_intermediate_size=converter.decoder_ff_intermediate_size,
        num_experts=converter.num_experts
        if isinstance(converter, MixtralForCausalLMConverter)
        else None,
    )


@pytest.fixture
def spec_data(model_name: str, render_model_config: ModelConfig) -> Dict[str, Any]:
    param_specs = get_param_specs(model_name, "models", render_model_config)
    return param_specs
