# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli OPT Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import torch
from transformers import OPTConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import DECODER_PREFIX, DecoderOnlyConverter
from friendli.modules.converter.schema import ConvertInfo


class OPTForCausalLMConverter(DecoderOnlyConverter):
    """OPTForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if OPT architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if cast(OPTConfig, self.config).activation_function not in ["relu"]:
                raise NotSupportedCheckpointError(
                    invalid_option="'activation_function="
                    f"{cast(OPTConfig, self.config).activation_function}'",
                    valid_options=["relu"],
                )
            if not cast(OPTConfig, self.config).do_layer_norm_before is True:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'do_layer_norm_before={False}'",
                    valid_options=[True],
                )
            if (
                cast(OPTConfig, self.config).word_embed_proj_dim
                != cast(OPTConfig, self.config).hidden_size
            ):
                raise NotSupportedCheckpointError(
                    invalid_option="'word_embed_proj_dim"
                    f"({cast(OPTConfig, self.config).word_embed_proj_dim}) "
                    f"!= hidden_size({cast(OPTConfig, self.config).hidden_size})'",
                    valid_options=[
                        f"'word_embed_proj_dim({cast(OPTConfig, self.config).hidden_size}) "
                        f"== hidden_size({cast(OPTConfig, self.config).hidden_size})'"
                    ],
                )
            if cast(  # pylint: disable=protected-access
                OPTConfig, self.config
            )._remove_final_layer_norm:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'_remove_final_layer_norm={True}'",
                    valid_options=[False],
                )
            if not cast(OPTConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'tie_word_embeddings={False}'",
                    valid_options=[True],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def pos_embed_weight_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Positional embedding weight convert for OPT's decoder."""
        assert len(params) == 1
        pos_emb = params[0]
        pos_emb = pos_emb[2:, :]  # offset pos emb

        return pos_emb

    def qkv_weight_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """qkv_weight_reshape for OPT's attention layer."""
        qkv_weight = torch.cat(
            params,
            dim=0,
        )
        qkv_weight = qkv_weight.transpose(0, 1)
        return qkv_weight

    def qkv_bias_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """qkv_bias_reshape for OPT's attention layer."""
        qkv_bias = torch.cat(
            params,
            dim=0,
        )
        return qkv_bias

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(OPTConfig, self.config)

        logger.warn(
            "Since OPT uses absolute position embedding, 'max_length' cannot be "
            "larger than %d.",
            config.max_position_embeddings,
        )

        eos_token_id = self.get_eos_token_id()
        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.decoder_head_size,
            "num_heads": self.decoder_num_attention_heads,
            "num_layers": self.decoder_layer_num,
            "max_length": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "opt"

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in OPT."""
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
                            f"{layer_prefix}self_attn.q_proj.bias",
                            f"{layer_prefix}self_attn.k_proj.bias",
                            f"{layer_prefix}self_attn.v_proj.bias",
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/bias:0",
                        reshape_fn=self.qkv_bias_reshape,
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
                        param_names=[f"{layer_prefix}fc1.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc2.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc2.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}fc1.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn.out_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
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
                ]
            )
        return convert_info_list

    @property
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for non-transformer blocks in OPT."""
        return [
            ConvertInfo(
                param_names=["model.decoder.embed_tokens.weight"],
                data_type=self.data_type,
                converted_name="wte/weight:0",
                reshape_fn=self.token_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.decoder.embed_positions.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/wpe/weight:0",
                reshape_fn=self.pos_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.decoder.final_layer_norm.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/gamma:0",
                reshape_fn=self.ln_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.decoder.final_layer_norm.bias"],
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
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before OPT's transformer block number."""
        return "model.decoder.layers."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in OPT."""
        return cast(OPTConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in OPT."""
        return cast(OPTConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in OPT."""
        return cast(OPTConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in opt."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of OPT."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in codegen OPT."""
        return self.decoder_hidden_size * 4
