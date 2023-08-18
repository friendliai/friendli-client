# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Falcon Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import FalconConfig  # type: ignore[import]

from periflow.errors import NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import DecoderOnlyConverter
from periflow.modules.converter.interface import DECODER_PREFIX
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    convert_to_gpt_j_params,
    get_tensor_from_state_dict,
)


class FalconForCausalLMConverter(DecoderOnlyConverter):
    """FalconForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Falcon architectures' config can be converted to PeriFlow format."""
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

    def qkv_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """qkv_weight_convert for Falcon's attention layer."""
        qkv_weight = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[0]
        )

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

        return convert_tensor_to_np_array(qkv_weight, self.data_type)

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
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        if cast(FalconConfig, self.config).new_decoder_architecture:
            return "falcon"
        return "falcon-7b"

    @property
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in Falcon."""
        convert_dict = {
            "attn/c_attn/weight:0": (
                self.qkv_weight_convert,
                [".self_attention.query_key_value.weight"],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".self_attention.dense.weight"],
            ),
            "mlp/c_fc/weight:0": (
                self.linear_weight_convert,
                [".mlp.dense_h_to_4h.weight"],
            ),
            "mlp/c_proj/weight:0": (
                self.linear_weight_convert,
                [".mlp.dense_4h_to_h.weight"],
            ),
        }
        if cast(FalconConfig, self.config).new_decoder_architecture:
            convert_dict.update(
                {
                    "ln_1/gamma:0": (
                        self.ln_weight_convert,
                        [".ln_attn.weight"],
                    ),
                    "ln_1/beta:0": (
                        self.ln_bias_convert,
                        [".ln_attn.bias"],
                    ),
                    "ln_2/gamma:0": (
                        self.ln_weight_convert,
                        [".ln_mlp.weight"],
                    ),
                    "ln_2/beta:0": (
                        self.ln_bias_convert,
                        [".ln_mlp.bias"],
                    ),
                }
            )
        else:
            convert_dict.update(
                {
                    "ln_1/gamma:0": (
                        self.ln_weight_convert,
                        [".input_layernorm.weight"],
                    ),
                    "ln_1/beta:0": (
                        self.ln_bias_convert,
                        [".input_layernorm.bias"],
                    ),
                }
            )
        return convert_dict

    @property
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in Falcon."""
        return {
            "wte/weight:0": (
                self.token_embed_weight_convert,
                ["transformer.word_embeddings.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/gamma:0": (
                self.ln_weight_convert,
                ["transformer.ln_f.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/beta:0": (
                self.ln_weight_convert,
                ["transformer.ln_f.bias"],
            ),
            "head_fc/weight:0": (
                self.head_weight_convert,
                ["lm_head.weight"],
            ),
        }

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before the Falcon's transformer layer number."""
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
    def rotary_dim(self) -> int:
        """The rotary embedding dimesion of Falcon."""
        return self.decoder_head_size
