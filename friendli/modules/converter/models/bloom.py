# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Bloom Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import BloomConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import DECODER_PREFIX, DecoderOnlyConverter
from friendli.modules.converter.schema import ConvertInfo


class BloomForCausalLMConverter(DecoderOnlyConverter):
    """BloomForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Bloom architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if cast(BloomConfig, self.config).apply_residual_connection_post_layernorm:
                raise NotSupportedCheckpointError(
                    invalid_option="apply_residual_connection_post_layernorm=True",
                    valid_options=[False],
                )
            if cast(BloomConfig, self.config).slow_but_exact:
                raise NotSupportedCheckpointError(
                    invalid_option="slow_but_exact=True", valid_options=[False]
                )
            if not cast(BloomConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="tie_word_embeddings=False", valid_options=[True]
                )
            if cast(BloomConfig, self.config).layer_norm_epsilon != 1e-5:
                raise NotSupportedCheckpointError(
                    invalid_option="layer_norm_epsilon="
                    f"{cast(BloomConfig, self.config).layer_norm_epsilon}",
                    valid_options=[1e-5],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def qkv_weight_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """qkv_weight_reshape for Bloom's attention layer."""
        assert len(params) == 1
        qkv_weight = params[0]
        split_qkv_weight_list = torch.split(qkv_weight, self.decoder_head_size, dim=0)
        qkv_weight_list = [
            torch.cat(
                [
                    split_qkv_weight_list[j * 3 + i]
                    for j in range(self.decoder_num_attention_heads)
                ],
                dim=0,
            ).reshape(-1, self.decoder_hidden_size)
            for i in range(3)
        ]

        qkv_weight = torch.cat(qkv_weight_list, dim=0).transpose(0, 1)
        return qkv_weight

    def qkv_bias_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """qkv_bias_reshape for Bloom's attention layer."""
        assert len(params) == 1
        qkv_bias = params[0]
        split_qkv_bias_list = torch.split(qkv_bias, self.decoder_head_size, dim=0)
        qkv_bias_list = [
            torch.cat(
                [
                    split_qkv_bias_list[j * 3 + i]
                    for j in range(self.decoder_num_attention_heads)
                ],
                dim=0,
            )
            for i in range(3)
        ]

        qkv_bias = torch.cat(qkv_bias_list, dim=0)
        return qkv_bias

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(BloomConfig, self.config)

        logger.warn(
            "The 'max_length' field is left blank as it cannot be automatically configured. "
            "You must determine the 'max_length' according to your needs. The Bloom model does "
            "not rely on absolute position embeddings, allowing you to choose any "
            "suitable value."
        )

        eos_token_id = self.get_eos_token_id()
        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.decoder_head_size,
            "num_heads": self.decoder_num_attention_heads,
            "num_layers": self.decoder_layer_num,
            "max_length": "FILL ME",
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "bloom"

    @property
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for non-transformer blocks in Bloom."""
        return [
            ConvertInfo(
                param_names=["transformer.word_embeddings.weight"],
                data_type=self.data_type,
                converted_name="wte/weight:0",
                reshape_fn=self.token_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["transformer.word_embeddings_layernorm.weight"],
                data_type=self.data_type,
                converted_name="wte/ln/gamma:0",
                reshape_fn=self.ln_weight_reshape,
            ),
            ConvertInfo(
                param_names=["transformer.word_embeddings_layernorm.bias"],
                data_type=self.data_type,
                converted_name="wte/ln/beta:0",
                reshape_fn=self.ln_bias_reshape,
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
        ]

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in Bloom."""
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
                        param_names=[
                            f"{layer_prefix}self_attention.query_key_value.bias"
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/bias:0",
                        reshape_fn=self.qkv_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attention.dense.bias"],
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
                        param_names=[
                            f"{layer_prefix}self_attention.query_key_value.weight"
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/weight:0",
                        reshape_fn=self.qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attention.dense.weight"],
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
        """The layer name prefix used before Bloom's transformer block number."""
        return "transformer.h."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in Bloom."""
        return cast(BloomConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """Return the hidden size in Bloom."""
        return cast(BloomConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in Bloom."""
        return cast(BloomConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in bloom."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The size of each attention head in Bloom."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in Bloom MLP."""
        return self.decoder_hidden_size * 4
