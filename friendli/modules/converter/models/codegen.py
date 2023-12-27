# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli CodeGen Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import torch
from transformers import CodeGenConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import (
    DECODER_PREFIX,
    SUPPORTED_GELU_FAMILY,
    DecoderOnlyConverter,
)
from friendli.modules.converter.interface import RotaryEmbeddingConversionInterface
from friendli.modules.converter.schema import ConvertInfo


class CodegenForCausalLMConverter(
    DecoderOnlyConverter, RotaryEmbeddingConversionInterface
):
    """CodegenForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if CodeGen architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if (
                cast(CodeGenConfig, self.config).activation_function
                not in SUPPORTED_GELU_FAMILY
            ):
                raise NotSupportedCheckpointError(
                    invalid_option="'activation_function="
                    f"{cast(CodeGenConfig, self.config).activation_function}'",
                    valid_options=SUPPORTED_GELU_FAMILY,
                )
            if cast(CodeGenConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if cast(CodeGenConfig, self.config).layer_norm_epsilon != 1e-5:
                raise NotSupportedCheckpointError(
                    invalid_option="'layer_norm_epsilon="
                    f"{cast(CodeGenConfig, self.config).layer_norm_epsilon}'",
                    valid_options=[1e-5],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def qkv_weight_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """qkv_weight_reshape for CodeGen's attention layer."""
        assert len(params) == 1
        original_qkv_weight = params[0]
        reshaped_qkv_weight = original_qkv_weight.reshape(
            (4, original_qkv_weight.size(0) // 4, original_qkv_weight.size(1))
        )
        q_weight, v_weight, k_weight = torch.split(
            reshaped_qkv_weight, reshaped_qkv_weight.size(1) // 3, dim=1
        )
        q_weight = q_weight.reshape((-1, q_weight.size(2)))
        k_weight = k_weight.reshape((-1, k_weight.size(2)))
        v_weight = v_weight.reshape((-1, v_weight.size(2)))

        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        qkv_weight = qkv_weight.transpose(0, 1)

        return qkv_weight

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(CodeGenConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The CodeGen model does not rely on "
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
        """The list of conversion informations for non-transformer blocks in CodeGen."""
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
                reshape_fn=self.linear_bias_reshape,
            ),
        ]

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in CodeGen."""
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
                        param_names=[f"{layer_prefix}attn.qkv_proj.weight"],
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
        """The layer name prefix used before CodeGen's transformer block number."""
        return "transformer.h."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in CodeGen."""
        return cast(CodeGenConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in CodeGen."""
        return cast(CodeGenConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in CodeGen."""
        return cast(CodeGenConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in the codegen."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head siez of CodeGen."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in codegen MLP."""
        return self.decoder_hidden_size * 4

    @property
    def rotary_dim(self) -> int:
        """The rotary dim in CodeGen."""
        return cast(CodeGenConfig, self.config).rotary_dim

    @property
    def rotary_emb_base(self) -> float:
        """The rotary emb base in CodeGen."""
        return 10000.0
