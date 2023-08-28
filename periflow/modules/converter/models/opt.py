# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow OPT Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import OPTConfig  # type: ignore[import]

from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import DecoderOnlyConverter
from periflow.modules.converter.interface import DECODER_PREFIX
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    get_tensor_from_state_dict,
)


class OPTForCausalLMConverter(DecoderOnlyConverter):
    """OPTForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if OPT architectures' config can be converted to PeriFlow format."""
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

    def pos_embed_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Positional embedding weight convert for OPT's decoder."""
        assert len(per_layer_postfixes) == 1
        pos_emb = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        pos_emb = pos_emb[2:, :]  # offset pos emb

        return convert_tensor_to_np_array(pos_emb, self.data_type)

    def qkv_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],  # type: ignore[override]
    ) -> np.ndarray:
        """qkv_weight_convert for OPT's attention layer."""
        qkv_weight = torch.cat(
            [
                get_tensor_from_state_dict(state_dict, layer + postfix)
                for postfix in per_layer_postfixes
            ],
            dim=0,
        )
        qkv_weight = qkv_weight.transpose(0, 1)
        return convert_tensor_to_np_array(qkv_weight, self.data_type)

    def qkv_bias_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],  # type: ignore[override]
    ) -> np.ndarray:
        """qkv_bias_convert for OPT's attention layer."""
        qkv_bias = torch.cat(
            [
                get_tensor_from_state_dict(state_dict, layer + postfix)
                for postfix in per_layer_postfixes
            ],
            dim=0,
        )
        return convert_tensor_to_np_array(qkv_bias, self.data_type)

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
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in OPT."""
        return {
            "ln_1/gamma:0": (
                self.ln_weight_convert,
                [".self_attn_layer_norm.weight"],
            ),
            "ln_1/beta:0": (
                self.ln_bias_convert,
                [".self_attn_layer_norm.bias"],
            ),
            "attn/c_attn/weight:0": (
                self.qkv_weight_convert,
                [
                    ".self_attn.q_proj.weight",
                    ".self_attn.k_proj.weight",
                    ".self_attn.v_proj.weight",
                ],
            ),
            "attn/c_attn/bias:0": (
                self.qkv_bias_convert,
                [
                    ".self_attn.q_proj.bias",
                    ".self_attn.k_proj.bias",
                    ".self_attn.v_proj.bias",
                ],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".self_attn.out_proj.weight"],
            ),
            "attn/c_proj/bias:0": (
                self.linear_bias_convert,
                [".self_attn.out_proj.bias"],
            ),
            "ln_2/gamma:0": (
                self.ln_weight_convert,
                [".final_layer_norm.weight"],
            ),
            "ln_2/beta:0": (
                self.ln_bias_convert,
                [".final_layer_norm.bias"],
            ),
            "mlp/c_fc/weight:0": (
                self.linear_weight_convert,
                [".fc1.weight"],
            ),
            "mlp/c_fc/bias:0": (
                self.linear_bias_convert,
                [".fc1.bias"],
            ),
            "mlp/c_proj/weight:0": (
                self.linear_weight_convert,
                [".fc2.weight"],
            ),
            "mlp/c_proj/bias:0": (
                self.linear_bias_convert,
                [".fc2.bias"],
            ),
        }

    @property
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in OPT."""
        return {
            "wte/weight:0": (
                self.token_embed_weight_convert,
                ["model.decoder.embed_tokens.weight"],
            ),
            DECODER_PREFIX
            + "/wpe/weight:0": (
                self.pos_embed_weight_convert,
                ["model.decoder.embed_positions.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/gamma:0": (
                self.ln_weight_convert,
                ["model.decoder.final_layer_norm.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/beta:0": (
                self.ln_bias_convert,
                ["model.decoder.final_layer_norm.bias"],
            ),
            "head_fc/weight:0": (
                self.head_weight_convert,
                ["lm_head.weight"],
            ),
        }

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before OPT's transformer layer number."""
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
