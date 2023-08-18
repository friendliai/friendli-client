# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Blenderbot Checkpoint Converter."""

from __future__ import annotations

import math
from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import BlenderbotConfig  # type: ignore[import]

from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import (
    SUPPORTED_GELU_FAMILY,
    EncoderDecoderConverter,
)
from periflow.modules.converter.interface import DECODER_PREFIX, ENCODER_PREFIX
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    get_tensor_from_state_dict,
)


class BlenderbotConverter(EncoderDecoderConverter):
    """BlenderbotForConditionalGeneration Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Blenderbot architectures's config can be converted to PeriFlow format."""
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

    def token_embed_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Token_embed_weight_convert for Blenderbot's embedding layer."""
        assert len(per_layer_postfixes) == 1
        embed_dim = cast(BlenderbotConfig, self.config).d_model
        embed_scale = (
            math.sqrt(embed_dim)
            if cast(BlenderbotConfig, self.config).scale_embedding
            else 1.0
        )

        embed_weight = get_tensor_from_state_dict(state_dict, per_layer_postfixes[0])
        embed_weight = embed_weight * embed_scale
        return convert_tensor_to_np_array(embed_weight, self.data_type)

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
            "ff_intermediate_size": config.encoder_ffn_dim,
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
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in Blenderbot."""
        return {
            "wte/weight:0": (self.token_embed_weight_convert, ["model.shared.weight"]),
            "head_fc/weight:0": (self.head_weight_convert, ["model.shared.weight"]),
            ENCODER_PREFIX
            + "/wpe/weight:0": (
                self.pos_embed_weight_convert,
                ["model.encoder.embed_positions.weight"],
            ),
            DECODER_PREFIX
            + "/wpe/weight:0": (
                self.pos_embed_weight_convert,
                ["model.decoder.embed_positions.weight"],
            ),
            ENCODER_PREFIX
            + "/ln_f/gamma:0": (
                self.ln_weight_convert,
                ["model.encoder.layer_norm.weight"],
            ),
            ENCODER_PREFIX
            + "/ln_f/beta:0": (self.ln_bias_convert, ["model.encoder.layer_norm.bias"]),
            DECODER_PREFIX
            + "/ln_f/gamma:0": (
                self.ln_weight_convert,
                ["model.decoder.layer_norm.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/beta:0": (self.ln_bias_convert, ["model.decoder.layer_norm.bias"]),
        }

    @property
    def encoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in Blenderbot's encoder."""
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
            "mlp/c_fc/weight:0": (self.linear_weight_convert, [".fc1.weight"]),
            "mlp/c_fc/bias:0": (
                self.linear_bias_convert,
                [".fc1.bias"],
            ),
            "mlp/c_proj/weight:0": (self.linear_weight_convert, [".fc2.weight"]),
            "mlp/c_proj/bias:0": (
                self.linear_bias_convert,
                [".fc2.bias"],
            ),
        }

    @property
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in Blenderbot's decoder."""
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
                [".encoder_attn_layer_norm.weight"],
            ),
            "ln_2/beta:0": (
                self.ln_bias_convert,
                [".encoder_attn_layer_norm.bias"],
            ),
            "ln_3/gamma:0": (
                self.ln_weight_convert,
                [".final_layer_norm.weight"],
            ),
            "ln_3/beta:0": (
                self.ln_bias_convert,
                [".final_layer_norm.bias"],
            ),
            "cross_attn/c_attn/weight:0": (
                self.qkv_weight_convert,
                [
                    ".encoder_attn.q_proj.weight",
                    ".encoder_attn.k_proj.weight",
                    ".encoder_attn.v_proj.weight",
                ],
            ),
            "cross_attn/c_attn/bias:0": (
                self.qkv_bias_convert,
                [
                    ".encoder_attn.q_proj.bias",
                    ".encoder_attn.k_proj.bias",
                    ".encoder_attn.v_proj.bias",
                ],
            ),
            "cross_attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".encoder_attn.out_proj.weight"],
            ),
            "cross_attn/c_proj/bias:0": (
                self.linear_bias_convert,
                [".encoder_attn.out_proj.bias"],
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
    def encoder_layer_prefix(self) -> str:
        """The layer name prefix used before Blenderbot encoder's transformer layer number."""
        return "model.encoder.layers."

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before Blenderbot decoder's transformer layer number."""
        return "model.decoder.layers."

    @property
    def encoder_layer_num(self) -> int:
        """The number of transformer layers in Blenderbot encoder."""
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
    def decoder_layer_num(self) -> int:
        """The number of transformer layers in Blenderbot decoder."""
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
