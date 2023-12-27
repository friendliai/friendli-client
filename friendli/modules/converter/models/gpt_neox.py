# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli GPT NeoX Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import torch
from transformers import GPTNeoXConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import (
    DECODER_PREFIX,
    SUPPORTED_GELU_FAMILY,
    DecoderOnlyConverter,
)
from friendli.modules.converter.interface import RotaryEmbeddingConversionInterface
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.converter.utils import convert_to_gpt_j_params


class GPTNeoXForCausalLMConverter(
    DecoderOnlyConverter, RotaryEmbeddingConversionInterface
):
    """GPTNeoXForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if GPTNeoX architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if cast(GPTNeoXConfig, self.config).hidden_act not in SUPPORTED_GELU_FAMILY:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'hidden_act={cast(GPTNeoXConfig, self.config).hidden_act}'",
                    valid_options=SUPPORTED_GELU_FAMILY,
                )
            if not cast(GPTNeoXConfig, self.config).use_parallel_residual:
                raise NotSupportedCheckpointError(
                    invalid_option="'use_parallel_residual=False'",
                    valid_options=[True],
                )
            if cast(GPTNeoXConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if cast(GPTNeoXConfig, self.config).layer_norm_eps != 1e-5:
                raise NotSupportedCheckpointError(
                    invalid_option="'layer_norm_eps="
                    f"{cast(GPTNeoXConfig, self.config).layer_norm_eps}'",
                    valid_options=[1e-5],
                )
            if cast(GPTNeoXConfig, self.config).rotary_emb_base != 10000:
                raise NotSupportedCheckpointError(
                    invalid_option=(
                        f"'rotary_emb_base={cast(GPTNeoXConfig, self.config).rotary_emb_base}'"
                    ),
                    valid_options=[10000],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def qkv_weight_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """qkv_weight_reshape for GPTNeoX's attention layer."""
        assert len(params) == 1
        qkv_weight = params[0]
        qkv_weight = qkv_weight.reshape(
            self.decoder_num_attention_heads,
            3,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )

        q_weight = qkv_weight[:, 0].reshape(
            self.decoder_num_attention_heads,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )
        k_weight = qkv_weight[:, 1].reshape(
            self.decoder_num_attention_heads,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )
        v_weight = qkv_weight[:, 2].reshape(
            self.decoder_num_attention_heads * self.decoder_head_size,
            self.decoder_hidden_size,
        )

        q_weight = convert_to_gpt_j_params(param=q_weight, rotary_dim=self.rotary_dim)
        k_weight = convert_to_gpt_j_params(param=k_weight, rotary_dim=self.rotary_dim)
        q_weight = q_weight.reshape(
            self.decoder_num_attention_heads * self.decoder_head_size,
            self.decoder_hidden_size,
        )
        k_weight = k_weight.reshape(
            self.decoder_num_attention_heads * self.decoder_head_size,
            self.decoder_hidden_size,
        )

        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        qkv_weight = qkv_weight.transpose(0, 1)

        return qkv_weight

    def qkv_bias_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """qkv_bias_reshape for GPTNeoX's attention layer."""
        assert len(params) == 1
        qkv_bias = params[0]
        qkv_bias = qkv_bias.reshape(
            self.decoder_num_attention_heads,
            3,
            self.decoder_head_size,
        )

        q_bias = qkv_bias[:, 0].reshape(
            self.decoder_num_attention_heads, self.decoder_head_size
        )
        k_bias = qkv_bias[:, 1].reshape(
            self.decoder_num_attention_heads, self.decoder_head_size
        )
        v_bias = qkv_bias[:, 2].reshape(
            self.decoder_num_attention_heads * self.decoder_head_size
        )

        q_bias = convert_to_gpt_j_params(q_bias, self.rotary_dim).flatten()
        k_bias = convert_to_gpt_j_params(k_bias, self.rotary_dim).flatten()

        qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
        return qkv_bias

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(GPTNeoXConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The GPTNeoX model does not rely on "
            "absolute position embeddings, allowing you to choose any suitable value.",
            config.max_position_embeddings,
        )

        eos_token_id = self.get_eos_token_id()
        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.decoder_head_size,
            "rotary_dim": self.rotary_dim,
            "num_heads": self.decoder_num_attention_heads,
            "num_layers": self.decoder_layer_num,
            "max_length": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
            "rope_theta": self.rotary_emb_base,
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "gpt-neox"

    @property
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for non-transformer blocks in GPTNeoX."""
        return [
            ConvertInfo(
                param_names=["gpt_neox.embed_in.weight"],
                data_type=self.data_type,
                converted_name="wte/weight:0",
                reshape_fn=self.token_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["gpt_neox.final_layer_norm.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/gamma:0",
                reshape_fn=self.ln_weight_reshape,
            ),
            ConvertInfo(
                param_names=["gpt_neox.final_layer_norm.bias"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/beta:0",
                reshape_fn=self.ln_bias_reshape,
            ),
            ConvertInfo(
                param_names=["embed_out.weight"],
                data_type=self.data_type,
                converted_name="head_fc/weight:0",
                reshape_fn=self.head_weight_reshape,
            ),
        ]

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in GPTNeoX."""
        convert_info_list = []
        for i in range(self.decoder_layer_num):
            layer_prefix = f"{self.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}input_layernorm.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}input_layernorm.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attention.query_key_value.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/bias:0",
                        reshape_fn=self.qkv_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attention.dense.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}post_attention_layernorm.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}post_attention_layernorm.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.dense_h_to_4h.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.dense_4h_to_h.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attention.query_key_value.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/weight:0",
                        reshape_fn=self.qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attention.dense.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.dense_h_to_4h.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.dense_4h_to_h.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                ]
            )
        return convert_info_list

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before GPTNeoX's transformer block number."""
        return "gpt_neox.layers."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in GPTNeoX."""
        return cast(GPTNeoXConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in GPTNeoX."""
        return cast(GPTNeoXConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in GPTNeoX."""
        return cast(GPTNeoXConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in gpt_neox."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head siez of GPTNeoX."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in codegen MLP."""
        return self.decoder_hidden_size * 4

    @property
    def rotary_dim(self) -> int:
        """The rotary embedding dimesion of GPTNeoX."""
        return int(self.decoder_head_size * cast(GPTNeoXConfig, self.config).rotary_pct)

    @property
    def rotary_emb_base(self) -> float:
        """The rotary embedding base of GPTNeoX."""
        return float(cast(GPTNeoXConfig, self.config).rotary_emb_base)
