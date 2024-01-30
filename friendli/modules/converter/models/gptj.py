# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli GPTJ Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import torch
from transformers import GPTJConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import (
    DECODER_PREFIX,
    SUPPORTED_GELU_FAMILY,
    DecoderOnlyConverter,
    DecoderOnlyLoraConverter,
)
from friendli.modules.converter.interface import RotaryEmbeddingConversionInterface
from friendli.modules.converter.schema import ConvertInfo


class GPTJForCausalLMLoraConverter(DecoderOnlyLoraConverter):
    """GPTJForCausalLM LoRA Converter Class."""

    @property
    def adapter_target_modules(self) -> List[str]:
        """Return the target modules that LoRA applies to."""
        return ["query", "value"]

    @property
    def adapter_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for LoRA adapter modules in GPTJ."""
        convert_info_list = []
        for i in range(self.converter.decoder_layer_num):
            layer_prefix = f"{self.converter.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/lora/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}attn.q_proj.lora_A.default.weight"
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}query_A/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}attn.q_proj.lora_B.default.weight"
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}query_B/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}attn.v_proj.lora_A.default.weight"
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}value_A/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}attn.v_proj.lora_B.default.weight"
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}value_B/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                ]
            )
        return convert_info_list


class GPTJForCausalLMConverter(
    DecoderOnlyConverter, RotaryEmbeddingConversionInterface
):
    """GPTJForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if GPTJ architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if (
                cast(GPTJConfig, self.config).activation_function
                not in SUPPORTED_GELU_FAMILY
            ):
                raise NotSupportedCheckpointError(
                    invalid_option="'activation_function="
                    f"{cast(GPTJConfig, self.config).activation_function}'",
                    valid_options=SUPPORTED_GELU_FAMILY,
                )
            if cast(GPTJConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if cast(GPTJConfig, self.config).layer_norm_epsilon != 1e-5:
                raise NotSupportedCheckpointError(
                    invalid_option="'layer_norm_epsilon="
                    f"{cast(GPTJConfig, self.config).layer_norm_epsilon}'",
                    valid_options=[1e-5],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def qkv_weight_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """qkv_weight_reshape for GPTJ's attention layer."""
        assert len(params) == 3
        qkv_weight = torch.cat(
            params,
            dim=0,
        )
        qkv_weight = qkv_weight.transpose(0, 1)
        return qkv_weight

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(GPTJConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The GPTJ model does not rely on "
            "absolute position embeddings, allowing you to choose any suitable value.",
            config.n_positions,
        )

        eos_token_id = self.get_eos_token_id()
        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.decoder_head_size,
            "rotary_dim": self.rotary_dim,
            "num_heads": self.decoder_num_attention_heads,
            "num_layers": self.decoder_layer_num,
            "max_length": config.n_positions,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
            "rope_theta": self.rotary_emb_base,
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "gpt-j"

    @property
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for non-transformer blocks in GPTJ."""
        return [
            ConvertInfo(
                param_names=["transformer.wte.weight"],
                data_type=self.data_type,
                converted_name="wte/weight:0",
                reshape_fn=self.token_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["transformer.ln_f.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/gamma:0",
                reshape_fn=self.ln_weight_reshape,
            ),
            ConvertInfo(
                param_names=["transformer.ln_f.bias"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/beta:0",
                reshape_fn=self.ln_bias_reshape,
            ),
            ConvertInfo(
                param_names=["lm_head.weight"],
                data_type=self.data_type,
                converted_name="head_fc/weight:0",
                reshape_fn=self.head_weight_reshape,
            ),
            ConvertInfo(
                param_names=["lm_head.bias"],
                data_type=self.data_type,
                converted_name="head_fc/bias:0",
                reshape_fn=self.head_weight_reshape,
            ),
        ]

    @property
    def decoder_convert_info_list(self) -> List[ConvertInfo]:
        """The list of conversion informations for transformer modules in GPTJ."""
        convert_info_list = []
        for i in range(self.decoder_layer_num):
            layer_prefix = f"{self.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ln_1.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ln_1.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.fc_in.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.fc_out.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}attn.q_proj.weight",
                            f"{layer_prefix}attn.k_proj.weight",
                            f"{layer_prefix}attn.v_proj.weight",
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/weight:0",
                        reshape_fn=self.qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn.out_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.fc_in.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.fc_out.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                ]
            )

        return convert_info_list

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before GPTJ's transformer module number."""
        return "transformer.h."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in GPTJ."""
        return cast(GPTJConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in GPTJ."""
        return cast(GPTJConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in GPTJ."""
        return cast(GPTJConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in gpt-j."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head siez of GPTJ."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in codegen MLP."""
        return self.decoder_hidden_size * 4

    @property
    def rotary_dim(self) -> int:
        """The rotary dim in GPTJ."""
        return cast(GPTJConfig, self.config).rotary_dim

    @property
    def rotary_emb_base(self) -> float:
        """The rotary emb base in GPTJ."""
        return 10000.0
