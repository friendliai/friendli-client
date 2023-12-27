# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Falcon Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import torch
from transformers import FalconConfig  # type: ignore[import]

from friendli.errors import NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import DECODER_PREFIX, DecoderOnlyConverter
from friendli.modules.converter.interface import RotaryEmbeddingConversionInterface
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.converter.utils import convert_to_gpt_j_params


class FalconForCausalLMConverter(
    DecoderOnlyConverter, RotaryEmbeddingConversionInterface
):
    """FalconForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Falcon architectures' config can be converted to Friendli format."""
        super().check_config()
        config = cast(FalconConfig, self.config)

        if config.layer_norm_epsilon != 1e-5:
            raise NotSupportedCheckpointError(
                invalid_option=f"'layer_norm_epsilon={config.layer_norm_epsilon}'",
                valid_options=[1e-5],
            )

        if config.alibi:
            raise NotSupportedCheckpointError(
                invalid_option=f"'alibi'={config.alibi}'",
                valid_options=[False],
            )

        if not config.rotary:
            raise NotSupportedCheckpointError(
                invalid_option=f"'rotary'={config.rotary}'",
                valid_options=[True],
            )

        if config.bias:
            raise NotSupportedCheckpointError(
                invalid_option=f"'bias'={config.bias}'",
                valid_options=[False],
            )

        if not config.new_decoder_architecture and not config.parallel_attn:
            raise NotSupportedCheckpointError(
                invalid_option=(
                    f"'new_decoder_architecture'={config.new_decoder_architecture}"
                    f"'parallel_attn'={config.parallel_attn}"
                ),
                valid_options=[
                    "'new_decoder_architecture'=True",
                    "'new_decoder_architecture'=False, 'parallel_attn'=True",
                ],
            )

    def qkv_weight_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """qkv_weight_reshape for Falcon's attention layer."""
        assert len(params) == 1
        qkv_weight = params[0]

        num_queries_per_kv = (
            self.decoder_num_attention_heads // self.decoder_num_kv_attention_heads
        )

        qkv_weight = qkv_weight.reshape(
            self.decoder_num_kv_attention_heads,
            num_queries_per_kv + 2,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )

        q_weight = qkv_weight[:, :num_queries_per_kv].reshape(
            self.decoder_num_kv_attention_heads * num_queries_per_kv,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )
        k_weight = qkv_weight[:, [-2]].reshape(
            self.decoder_num_kv_attention_heads,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )
        v_weight = qkv_weight[:, [-1]].reshape(
            self.decoder_num_kv_attention_heads * self.decoder_head_size,
            self.decoder_hidden_size,
        )

        q_weight = convert_to_gpt_j_params(q_weight, self.rotary_dim)
        k_weight = convert_to_gpt_j_params(k_weight, self.rotary_dim)

        q_weight = q_weight.reshape(
            self.decoder_num_kv_attention_heads
            * num_queries_per_kv
            * self.decoder_head_size,
            self.decoder_hidden_size,
        )
        k_weight = k_weight.reshape(
            self.decoder_num_kv_attention_heads * self.decoder_head_size,
            self.decoder_hidden_size,
        )

        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        qkv_weight = qkv_weight.transpose(0, 1)

        return qkv_weight

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(FalconConfig, self.config)

        logger.warn(
            "The 'max_length' field is left blank as it cannot be automatically configured. "
            "You must determine the 'max_length' according to your needs. The Falcon model does "
            "not rely on absolute position embeddings, allowing you to choose any "
            "suitable value."
        )

        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.decoder_head_size,
            "rotary_dim": self.rotary_dim,
            "num_heads": self.decoder_num_attention_heads,
            "num_kv_heads": self.decoder_num_kv_attention_heads,
            "num_layers": self.decoder_layer_num,
            "max_length": "FILL ME",
            "vocab_size": config.vocab_size,
            "eos_token": self.get_eos_token_id() or "FILL ME",
            "rope_theta": self.rotary_emb_base,
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        if cast(FalconConfig, self.config).new_decoder_architecture:
            return "falcon"
        return "falcon-7b"

    @property
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for non-transformer blocks in Falcon."""
        return [
            ConvertInfo(
                param_names=["transformer.word_embeddings.weight"],
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
        ]

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in Falcon."""
        convert_info_list = []
        for i in range(self.decoder_layer_num):
            layer_prefix = f"{self.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"

            convert_info_list.extend(
                [
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

            if cast(FalconConfig, self.config).new_decoder_architecture:
                convert_info_list.extend(
                    [
                        ConvertInfo(
                            param_names=[f"{layer_prefix}ln_attn.weight"],
                            data_type=self.data_type,
                            converted_name=f"{converted_prefix}ln_1/gamma:0",
                            reshape_fn=self.ln_weight_reshape,
                        ),
                        ConvertInfo(
                            param_names=[f"{layer_prefix}ln_attn.bias"],
                            data_type=self.data_type,
                            converted_name=f"{converted_prefix}ln_1/beta:0",
                            reshape_fn=self.ln_bias_reshape,
                        ),
                        ConvertInfo(
                            param_names=[f"{layer_prefix}ln_mlp.weight"],
                            data_type=self.data_type,
                            converted_name=f"{converted_prefix}ln_2/gamma:0",
                            reshape_fn=self.ln_weight_reshape,
                        ),
                        ConvertInfo(
                            param_names=[f"{layer_prefix}ln_mlp.bias"],
                            data_type=self.data_type,
                            converted_name=f"{converted_prefix}ln_2/beta:0",
                            reshape_fn=self.ln_bias_reshape,
                        ),
                    ]
                )
            else:
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
                    ]
                )

        return convert_info_list

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before the Falcon's transformer block number."""
        return "transformer.h."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in Falcon."""
        return cast(FalconConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in Falcon."""
        return cast(FalconConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in Falcon."""
        return cast(FalconConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in Falcon."""
        config = cast(FalconConfig, self.config)

        if config.new_decoder_architecture:
            if config.num_kv_heads is not None:
                return config.num_kv_heads
            return config.num_attention_heads

        if config.multi_query:
            return 1

        if config.num_kv_heads is not None:
            return config.num_kv_heads
        return config.num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of Falcon."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in falcon MLP."""
        return self.decoder_hidden_size * 4

    @property
    def rotary_dim(self) -> int:
        """The rotary embedding dimesion of Falcon."""
        return self.decoder_head_size

    @property
    def rotary_emb_base(self) -> float:
        """The rotary embedding base of Falcon."""
        return cast(FalconConfig, self.config).rope_theta
