# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Bloom Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import BloomConfig  # type: ignore[import]

from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import DecoderOnlyConverter
from periflow.modules.converter.interface import DECODER_PREFIX
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    get_tensor_from_state_dict,
)


class BloomForCausalLMConverter(DecoderOnlyConverter):
    """BloomForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Bloom architectures' config can be converted to PeriFlow format."""
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

    def qkv_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """qkv_weight_convert for Bloom's attention layer."""
        assert len(per_layer_postfixes) == 1
        qkv_weight = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[0]
        )
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
        return convert_tensor_to_np_array(qkv_weight, self.data_type)

    def qkv_bias_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: str,  # type: ignore[override]
    ) -> np.ndarray:
        """qkv_bias_convert for Bloom's attention layer."""
        assert len(per_layer_postfixes) == 1
        qkv_bias = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[0]
        )
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
        return convert_tensor_to_np_array(qkv_bias, self.data_type)

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
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in Bloom."""
        return {
            "wte/weight:0": (
                self.token_embed_weight_convert,
                ["transformer.word_embeddings.weight"],
            ),
            "wte/ln/gamma:0": (
                self.ln_weight_convert,
                ["transformer.word_embeddings_layernorm.weight"],
            ),
            "wte/ln/beta:0": (
                self.ln_bias_convert,
                ["transformer.word_embeddings_layernorm.bias"],
            ),
            DECODER_PREFIX
            + "/ln_f/gamma:0": (self.ln_weight_convert, ["transformer.ln_f.weight"]),
            DECODER_PREFIX
            + "/ln_f/beta:0": (self.ln_bias_convert, ["transformer.ln_f.bias"]),
        }

    @property
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in Bloom."""
        return {
            "ln_1/gamma:0": (
                self.ln_weight_convert,
                [".input_layernorm.weight"],
            ),
            "ln_1/beta:0": (
                self.ln_bias_convert,
                [".input_layernorm.bias"],
            ),
            "attn/c_attn/weight:0": (
                self.qkv_weight_convert,
                [".self_attention.query_key_value.weight"],
            ),
            "attn/c_attn/bias:0": (
                self.qkv_bias_convert,
                [".self_attention.query_key_value.bias"],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".self_attention.dense.weight"],
            ),
            "attn/c_proj/bias:0": (
                self.linear_bias_convert,
                [".self_attention.dense.bias"],
            ),
            "ln_2/gamma:0": (
                self.ln_weight_convert,
                [".post_attention_layernorm.weight"],
            ),
            "ln_2/beta:0": (
                self.ln_bias_convert,
                [".post_attention_layernorm.bias"],
            ),
            "mlp/c_fc/weight:0": (
                self.linear_weight_convert,
                [".mlp.dense_h_to_4h.weight"],
            ),
            "mlp/c_fc/bias:0": (
                self.linear_bias_convert,
                [".mlp.dense_h_to_4h.bias"],
            ),
            "mlp/c_proj/weight:0": (
                self.linear_weight_convert,
                [".mlp.dense_4h_to_h.weight"],
            ),
            "mlp/c_proj/bias:0": (
                self.linear_bias_convert,
                [".mlp.dense_4h_to_h.bias"],
            ),
        }

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before Bloom's transformer layer number."""
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
