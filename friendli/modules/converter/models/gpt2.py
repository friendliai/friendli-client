# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli GPT2 Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import torch
from transformers import GPT2Config  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import (
    DECODER_PREFIX,
    SUPPORTED_GELU_FAMILY,
    DecoderOnlyConverter,
)
from friendli.modules.converter.schema import ConvertInfo


class GPT2LMHeadModelConverter(DecoderOnlyConverter):
    """GPT2LMHeadModel Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if GPT2 architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if (
                cast(GPT2Config, self.config).activation_function
                not in SUPPORTED_GELU_FAMILY
            ):
                raise NotSupportedCheckpointError(
                    invalid_option="'activation_function="
                    f"{cast(GPT2Config, self.config).activation_function}'",
                    valid_options=SUPPORTED_GELU_FAMILY,
                )
            if cast(GPT2Config, self.config).scale_attn_by_inverse_layer_idx:
                raise NotSupportedCheckpointError(
                    invalid_option="'scale_attn_by_inverse_layer_idx=True'",
                    valid_options=[False],
                )
            if not cast(GPT2Config, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=False'",
                    valid_options=[True],
                )
            if cast(GPT2Config, self.config).layer_norm_epsilon != 1e-5:
                raise NotSupportedCheckpointError(
                    invalid_option="'layer_norm_epsilon="
                    f"{cast(GPT2Config, self.config).layer_norm_epsilon}'",
                    valid_options=[1e-5],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(GPT2Config, self.config)

        logger.warn(
            "Since GPT2 uses absolute position embedding, 'max_length' cannot be "
            "larger than %d.",
            config.n_positions,
        )

        eos_token_id = self.get_eos_token_id()
        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.decoder_head_size,
            "num_heads": self.decoder_num_attention_heads,
            "num_layers": self.decoder_layer_num,
            "max_length": config.n_positions,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "gpt"

    @property
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for non-transformer blocks in GPT2."""
        return [
            ConvertInfo(
                param_names=["transformer.wte.weight"],
                data_type=self.data_type,
                converted_name="wte/weight:0",
                reshape_fn=self.token_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["transformer.wpe.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/wpe/weight:0",
                reshape_fn=self.pos_embed_weight_reshape,
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

    def linear_weight_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape linear weight in GPT2, which does not need weight transpose."""
        assert len(params) == 1
        return params[0]

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in GPT2."""
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
                        param_names=[f"{layer_prefix}attn.c_attn.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn.c_proj.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ln_2.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ln_2.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.c_fc.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.c_proj.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn.c_attn.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn.c_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.c_fc.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.c_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                ]
            )
        return convert_info_list

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before GPT2's transformer block number."""
        return "transformer.h."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in GPT2."""
        return cast(GPT2Config, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in GPT2."""
        return cast(GPT2Config, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in GPT2."""
        return cast(GPT2Config, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in gpt2."""
        return self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in codegen MLP."""
        return self.decoder_hidden_size * 4

    @property
    def decoder_head_size(self) -> int:
        """The head siez of GPT2."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads
