# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow LLaMAa Checkpoint Converter."""


from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import LlamaConfig  # type: ignore[import]

from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import DecoderOnlyConverter
from periflow.modules.converter.interface import DECODER_PREFIX
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    convert_to_gpt_j_params,
    get_tensor_from_state_dict,
)


class LlamaForCausalLMConverter(DecoderOnlyConverter):
    """LlamaForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if LLaMA architectures' config can be converted to PeriFlow format."""
        super().check_config()
        try:
            if cast(LlamaConfig, self.config).hidden_act not in ["silu"]:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'hidden_act={cast(LlamaConfig, self.config).hidden_act}'",
                    valid_options=["silu"],
                )
            if cast(LlamaConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if cast(LlamaConfig, self.config).rms_norm_eps not in (1e-5, 1e-6):
                raise NotSupportedCheckpointError(
                    invalid_option=f"'rms_norm_eps={cast(LlamaConfig, self.config).rms_norm_eps}'",
                    valid_options=[1e-5, 1e-6],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def qkv_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],  # type: ignore[override]
    ) -> np.ndarray:
        """qkv_weight_convert for LLaMA's attention layer."""
        assert len(per_layer_postfixes) == 3
        q_weight = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[0]
        )
        k_weight = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[1]
        )
        v_weight = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[2]
        )

        q_weight = q_weight.reshape(
            self.decoder_num_attention_heads,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )
        k_weight = k_weight.reshape(
            self.decoder_num_kv_attention_heads,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )
        q_weight = convert_to_gpt_j_params(q_weight, self.rotary_dim)
        k_weight = convert_to_gpt_j_params(k_weight, self.rotary_dim)
        q_weight = q_weight.reshape(
            self.decoder_num_attention_heads * self.decoder_head_size,
            self.decoder_hidden_size,
        )
        k_weight = k_weight.reshape(
            self.decoder_num_kv_attention_heads * self.decoder_head_size,
            self.decoder_hidden_size,
        )

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        qkv_weight = qkv_weight.transpose(0, -1)
        return convert_tensor_to_np_array(qkv_weight, self.data_type)

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(LlamaConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The Llama model does not rely on "
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
            "num_kv_heads": self.decoder_num_kv_attention_heads,
            "num_layers": self.decoder_layer_num,
            "ff_intermediate_size": config.intermediate_size,
            "max_length": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "llama"

    @property
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in LLaMA."""
        return {
            "ln_1/gamma:0": (
                self.ln_weight_convert,
                [".input_layernorm.weight"],
            ),
            "attn/c_attn/weight:0": (
                self.qkv_weight_convert,
                [
                    ".self_attn.q_proj.weight",
                    ".self_attn.k_proj.weight",
                    ".self_attn.v_proj.weight",
                ],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".self_attn.o_proj.weight"],
            ),
            "ln_2/gamma:0": (
                self.ln_weight_convert,
                [".post_attention_layernorm.weight"],
            ),
            "mlp/c_gate/weight:0": (
                self.linear_weight_convert,
                [".mlp.gate_proj.weight"],
            ),
            "mlp/c_fc/weight:0": (
                self.linear_weight_convert,
                [".mlp.up_proj.weight"],
            ),
            "mlp/c_proj/weight:0": (
                self.linear_weight_convert,
                [".mlp.down_proj.weight"],
            ),
        }

    @property
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in LLaMA."""
        return {
            "wte/weight:0": (
                self.token_embed_weight_convert,
                ["model.embed_tokens.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/gamma:0": (
                self.ln_weight_convert,
                ["model.norm.weight"],
            ),
            "head_fc/weight:0": (
                self.head_weight_convert,
                ["lm_head.weight"],
            ),
        }

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before LLaMA's transformer layer number."""
        return "model.layers."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in LLaMA."""
        return cast(LlamaConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in LLaMA."""
        return cast(LlamaConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in LLaMA."""
        return cast(LlamaConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in LLaMA."""
        config = cast(LlamaConfig, self.config)
        if config.num_key_value_heads is None:
            return self.decoder_num_attention_heads
        return config.num_key_value_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of LLaMA."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def rotary_dim(self) -> int:
        """The rotary embedding dimension of LLaMA."""
        return self.decoder_head_size
