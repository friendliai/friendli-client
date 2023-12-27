# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Blenderbot Checkpoint Converter."""

from __future__ import annotations

import math
from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import BlenderbotConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import (
    DECODER_PREFIX,
    ENCODER_PREFIX,
    SUPPORTED_GELU_FAMILY,
    EncoderDecoderConverter,
)
from friendli.modules.converter.schema import ConvertInfo


class BlenderbotConverter(EncoderDecoderConverter):
    """BlenderbotForConditionalGeneration Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Blenderbot architectures's config can be converted to Friendli format."""
        super().check_config()
        config = cast(BlenderbotConfig, self.config)
        try:
            if config.activation_function not in SUPPORTED_GELU_FAMILY:
                raise NotSupportedCheckpointError(
                    invalid_option="'activation_function="
                    f"{cast(BlenderbotConfig, self.config).activation_function}'",
                    valid_options=SUPPORTED_GELU_FAMILY,
                )
            if not config.tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=False'",
                    valid_options=[True],
                )
            if self.encoder_num_attention_heads != self.decoder_num_attention_heads:
                raise NotSupportedCheckpointError(
                    invalid_option=(
                        f"encoder_num_attention_heads={self.encoder_num_attention_heads} "
                        f"decoder_num_attention_heads={self.decoder_num_attention_heads}"
                    ),
                    valid_options=[
                        "encoder_num_attention_heads == decoder_num_attention_heads"
                    ],
                )
            if config.decoder_ffn_dim != config.encoder_ffn_dim:
                raise NotSupportedCheckpointError(
                    invalid_option=(
                        f"encoder_ffn_dim={config.encoder_ffn_dim} "
                        f"decoder_ffn_dim={config.decoder_ffn_dim}"
                    ),
                    valid_options=["encoder_ffn_dim == decoder_ffn_dim"],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def token_embed_weight_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Reshape token embedding weight for Blenderbot's embedding layer."""
        assert len(params) == 1
        embed_dim = cast(BlenderbotConfig, self.config).d_model
        embed_scale = (
            math.sqrt(embed_dim)
            if cast(BlenderbotConfig, self.config).scale_embedding
            else 1.0
        )
        embed_weight = params[0]
        embed_weight = embed_weight * embed_scale
        return embed_weight

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(BlenderbotConfig, self.config)

        logger.warn(
            "Since Blenderbot uses absolute position embedding, 'max_input_length' and "
            "'max_output_length' cannot be larger than %d.",
            config.max_position_embeddings,
        )

        eos_token_id = self.get_eos_token_id()
        decoder_start_token_id = self.get_decoder_start_token_id()
        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.encoder_head_size,
            "num_heads": self.encoder_num_attention_heads,
            "hidden_size": self.encoder_hidden_size,
            "ff_intermediate_size": self.decoder_ff_intermediate_size,
            "num_encoder_layers": self.encoder_layer_num,
            "num_decoder_layers": self.decoder_layer_num,
            "max_input_length": config.max_position_embeddings,
            "max_output_length": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
            "decoder_start_token": (
                decoder_start_token_id
                if decoder_start_token_id is not None
                else "FILL ME"
            ),
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "blenderbot"

    @property
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for non-transformer blocks in Blenderbot."""
        return [
            ConvertInfo(
                param_names=["model.shared.weight"],
                data_type=self.data_type,
                converted_name="wte/weight:0",
                reshape_fn=self.token_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.shared.weight"],
                data_type=self.data_type,
                converted_name="head_fc/weight:0",
                reshape_fn=self.head_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.encoder.embed_positions.weight"],
                data_type=self.data_type,
                converted_name=f"{ENCODER_PREFIX}/wpe/weight:0",
                reshape_fn=self.pos_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.decoder.embed_positions.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/wpe/weight:0",
                reshape_fn=self.pos_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.encoder.layer_norm.weight"],
                data_type=self.data_type,
                converted_name=f"{ENCODER_PREFIX}/ln_f/gamma:0",
                reshape_fn=self.ln_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.encoder.layer_norm.bias"],
                data_type=self.data_type,
                converted_name=f"{ENCODER_PREFIX}/ln_f/beta:0",
                reshape_fn=self.ln_bias_reshape,
            ),
            ConvertInfo(
                param_names=["model.decoder.layer_norm.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/gamma:0",
                reshape_fn=self.ln_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.decoder.layer_norm.bias"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/beta:0",
                reshape_fn=self.ln_bias_reshape,
            ),
        ]

    @property
    def encoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in Blenderbot's encoder."""
        convert_info_list = []
        for i in range(self.encoder_layer_num):
            layer_prefix = f"{self.encoder_layer_prefix}{i}."
            converted_prefix = f"{ENCODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn_layer_norm.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn_layer_norm.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}self_attn.q_proj.weight",
                            f"{layer_prefix}self_attn.k_proj.weight",
                            f"{layer_prefix}self_attn.v_proj.weight",
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/weight:0",
                        reshape_fn=self.qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}self_attn.q_proj.bias",
                            f"{layer_prefix}self_attn.k_proj.bias",
                            f"{layer_prefix}self_attn.v_proj.bias",
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/bias:0",
                        reshape_fn=self.qkv_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn.out_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn.out_proj.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}final_layer_norm.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}final_layer_norm.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc1.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc1.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc2.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc2.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                ]
            )
        return convert_info_list

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in Blenderbot's decoder."""
        convert_info_list = []
        for i in range(self.decoder_layer_num):
            layer_prefix = f"{self.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn_layer_norm.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn_layer_norm.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}self_attn.q_proj.weight",
                            f"{layer_prefix}self_attn.k_proj.weight",
                            f"{layer_prefix}self_attn.v_proj.weight",
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/weight:0",
                        reshape_fn=self.qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}self_attn.q_proj.bias",
                            f"{layer_prefix}self_attn.k_proj.bias",
                            f"{layer_prefix}self_attn.v_proj.bias",
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/bias:0",
                        reshape_fn=self.qkv_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn.out_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn.out_proj.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}encoder_attn_layer_norm.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}encoder_attn_layer_norm.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}final_layer_norm.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_3/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}final_layer_norm.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_3/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}encoder_attn.q_proj.weight",
                            f"{layer_prefix}encoder_attn.k_proj.weight",
                            f"{layer_prefix}encoder_attn.v_proj.weight",
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}cross_attn/c_attn/weight:0",
                        reshape_fn=self.qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}encoder_attn.q_proj.bias",
                            f"{layer_prefix}encoder_attn.k_proj.bias",
                            f"{layer_prefix}encoder_attn.v_proj.bias",
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}cross_attn/c_attn/bias:0",
                        reshape_fn=self.qkv_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}encoder_attn.out_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}cross_attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}encoder_attn.out_proj.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}cross_attn/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc1.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc1.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc2.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc2.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                ]
            )
        return convert_info_list

    @property
    def encoder_layer_prefix(self) -> str:
        """The layer name prefix used before Blenderbot encoder's transformer block number."""
        return "model.encoder.layers."

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before Blenderbot decoder's transformer block number."""
        return "model.decoder.layers."

    @property
    def encoder_layer_num(self) -> int:
        """The number of transformer blocks in Blenderbot encoder."""
        return cast(BlenderbotConfig, self.config).encoder_layers

    @property
    def encoder_hidden_size(self) -> int:
        """The hidden size of Blenderbot encoder."""
        return cast(BlenderbotConfig, self.config).d_model

    @property
    def encoder_num_attention_heads(self) -> int:
        """The number of attention heads of Blenderbot encoder."""
        return cast(BlenderbotConfig, self.config).encoder_attention_heads

    @property
    def encoder_head_size(self) -> int:
        """The size of each attention head of Blenderbot encoder."""
        return self.encoder_hidden_size // self.encoder_num_attention_heads

    @property
    def encoder_ff_intermediate_size(self) -> int:
        """The intermediate of the linear layer in Blenderbot encoder's MLP."""
        return cast(BlenderbotConfig, self.config).encoder_ffn_dim

    @property
    def decoder_layer_num(self) -> int:
        """The number of transformer blocks in Blenderbot decoder."""
        return cast(BlenderbotConfig, self.config).decoder_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size of Blenderbot decoder."""
        return cast(BlenderbotConfig, self.config).d_model

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads of Blenderbot decoder."""
        return cast(BlenderbotConfig, self.config).decoder_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads of blenderbot decoder."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The size of each attention head of Blenderbot decoder."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate of the linear layer in Blenderbot decoder's MLP."""
        return cast(BlenderbotConfig, self.config).decoder_ffn_dim
