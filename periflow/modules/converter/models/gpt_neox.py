# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPT NeoX Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, cast

import numpy as np
import torch
from transformers import GPTNeoXConfig  # type: ignore[import]

from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import SUPPORTED_GELU_FAMILY, DecoderOnlyConverter
from periflow.modules.converter.interface import DECODER_PREFIX
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    convert_to_gpt_j_params,
    get_tensor_from_state_dict,
)


class GPTNeoXForCausalLMConverter(DecoderOnlyConverter):
    """GPTNeoXForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if GPTNeoX architectures' config can be converted to Periflow format."""
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

    def qkv_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: str,  # type: ignore[override]
    ) -> np.ndarray:
        """qkv_weight_convert for GPTNeoX's attention layer."""
        assert len(per_layer_postfixes) == 1
        qkv_weight = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[0]
        )
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

        return convert_tensor_to_np_array(qkv_weight, self.data_type)

    def qkv_bias_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: str,  # type: ignore[override]
    ) -> np.ndarray:
        """qkv_bias_convert for GPTNeoX's attention layer."""
        assert len(per_layer_postfixes) == 1
        qkv_bias = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[0]
        )
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
        return convert_tensor_to_np_array(qkv_bias, self.data_type)

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
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "gpt-neox"

    @property
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in GPTNeoX."""
        return {
            "wte/weight:0": (
                self.token_embed_weight_convert,
                ["gpt_neox.embed_in.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/gamma:0": (
                self.ln_weight_convert,
                ["gpt_neox.final_layer_norm.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/beta:0": (
                self.ln_bias_convert,
                ["gpt_neox.final_layer_norm.bias"],
            ),
            "head_fc/weight:0": (self.head_weight_convert, ["embed_out.weight"]),
        }

    @property
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in GPTNeoX."""
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
                [".attention.query_key_value.weight"],
            ),
            "attn/c_attn/bias:0": (
                self.qkv_bias_convert,
                [".attention.query_key_value.bias"],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".attention.dense.weight"],
            ),
            "attn/c_proj/bias:0": (
                self.linear_bias_convert,
                [".attention.dense.bias"],
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
        """The layer name prefix used before GPTNeoX's transformer layer number."""
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
    def rotary_dim(self) -> int:
        """The rotary embedding dimesion of GPTNeoX."""
        return int(self.decoder_head_size * cast(GPTNeoXConfig, self.config).rotary_pct)
