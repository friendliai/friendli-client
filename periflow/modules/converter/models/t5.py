# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow T5 Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import T5Config  # type: ignore[import]

from periflow.enums import CheckpointDataType
from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import EncoderDecoderConverter
from periflow.modules.converter.interface import DECODER_PREFIX, ENCODER_PREFIX
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    get_tensor_from_state_dict,
)


class T5Converter(EncoderDecoderConverter):
    """T5ForConditionalGeneration Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if T5 architectures' config can be converted to PeriFlow format."""
        super().check_config()
        try:
            if not (
                cast(T5Config, self.config).is_gated_act
                ^ cast(T5Config, self.config).tie_word_embeddings
            ):
                raise NotSupportedCheckpointError(
                    invalid_option=f"'is_gated_act={cast(T5Config, self.config).is_gated_act}'and "
                    f"'tie_word_embeddings={cast(T5Config, self.config).tie_word_embeddings}'",
                    valid_options=[
                        "'is_gated_act' and 'tie_word_embeddings' should be different."
                    ],
                )

            if cast(T5Config, self.config).layer_norm_epsilon != 1e-6:
                raise NotSupportedCheckpointError(
                    invalid_option="'layer_norm_epsilon="
                    f"{cast(T5Config, self.config).layer_norm_epsilon}'",
                    valid_options=[1e-6],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def _decoder_final_ln_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Special handle for T5."""
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])

        if cast(T5Config, self.config).tie_word_embeddings:
            param = param * (cast(T5Config, self.config).d_model ** -0.5)

        return convert_tensor_to_np_array(param, self.data_type)

    def pos_embed_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert positional embedding weights in T5."""
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        return convert_tensor_to_np_array(
            param=param, data_type=CheckpointDataType.FP32
        )

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(T5Config, self.config)

        logger.warn(
            "The 'max_input_length' and 'max_output_length' fields are left blank as "
            "they cannot be automatically configured. "
            "Determine the 'max_input_length' and 'max_output_length' according to your "
            "needs. The T5 model does not rely on absolute position embeddings, "
            "allowing you to choose any suitable value."
        )

        eos_token_id = self.get_eos_token_id()
        decoder_start_token_id = self.get_decoder_start_token_id()
        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.encoder_head_size,
            "num_heads": self.encoder_num_attention_heads,
            "hidden_size": self.encoder_hidden_size,
            "ff_intermediate_size": config.d_ff,
            "num_encoder_layers": self.encoder_layer_num,
            "num_decoder_layers": self.decoder_layer_num,
            "max_input_length": "FILL ME",
            "max_output_length": "FILL ME",
            "num_pos_emb_buckets": config.relative_attention_num_buckets,
            "max_pos_distance": config.relative_attention_max_distance,
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
        if cast(T5Config, self.config).is_gated_act:
            return "t5-v1_1"
        return "t5"

    @property
    def encoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in T5's encoder."""
        convert_dict = {
            "ln_1/gamma:0": (
                self.ln_weight_convert,
                [".layer.0.layer_norm.weight"],
            ),
            "ln_2/gamma:0": (
                self.ln_weight_convert,
                [".layer.1.layer_norm.weight"],
            ),
            "attn/c_attn/weight:0": (
                self.qkv_weight_convert,
                [
                    ".layer.0.SelfAttention.q.weight",
                    ".layer.0.SelfAttention.k.weight",
                    ".layer.0.SelfAttention.v.weight",
                ],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".layer.0.SelfAttention.o.weight"],
            ),
        }
        if cast(T5Config, self.config).is_gated_act:
            convert_dict.update(
                {
                    "mlp/c_gate/weight:0": (
                        self.linear_weight_convert,
                        [".layer.1.DenseReluDense.wi_0.weight"],
                    ),
                    "mlp/c_fc/weight:0": (
                        self.linear_weight_convert,
                        [".layer.1.DenseReluDense.wi_1.weight"],
                    ),
                    "mlp/c_proj/weight:0": (
                        self.linear_weight_convert,
                        [".layer.1.DenseReluDense.wo.weight"],
                    ),
                }
            )
        else:
            convert_dict.update(
                {
                    "mlp/c_fc/weight:0": (
                        self.linear_weight_convert,
                        [".layer.1.DenseReluDense.wi.weight"],
                    ),
                    "mlp/c_proj/weight:0": (
                        self.linear_weight_convert,
                        [".layer.1.DenseReluDense.wo.weight"],
                    ),
                }
            )
        return convert_dict

    @property
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in T5's decoder."""
        convert_dict = {
            "ln_1/gamma:0": (
                self.ln_weight_convert,
                [".layer.0.layer_norm.weight"],
            ),
            "ln_2/gamma:0": (
                self.ln_weight_convert,
                [".layer.1.layer_norm.weight"],
            ),
            "ln_3/gamma:0": (
                self.ln_weight_convert,
                [".layer.2.layer_norm.weight"],
            ),
            "attn/c_attn/weight:0": (
                self.qkv_weight_convert,
                [
                    ".layer.0.SelfAttention.q.weight",
                    ".layer.0.SelfAttention.k.weight",
                    ".layer.0.SelfAttention.v.weight",
                ],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".layer.0.SelfAttention.o.weight"],
            ),
            "cross_attn/c_attn/weight:0": (
                self.qkv_weight_convert,
                [
                    ".layer.1.EncDecAttention.q.weight",
                    ".layer.1.EncDecAttention.k.weight",
                    ".layer.1.EncDecAttention.v.weight",
                ],
            ),
            "cross_attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".layer.1.EncDecAttention.o.weight"],
            ),
        }
        if cast(T5Config, self.config).is_gated_act:
            convert_dict.update(
                {
                    "mlp/c_gate/weight:0": (
                        self.linear_weight_convert,
                        [".layer.2.DenseReluDense.wi_0.weight"],
                    ),
                    "mlp/c_fc/weight:0": (
                        self.linear_weight_convert,
                        [".layer.2.DenseReluDense.wi_1.weight"],
                    ),
                    "mlp/c_proj/weight:0": (
                        self.linear_weight_convert,
                        [".layer.2.DenseReluDense.wo.weight"],
                    ),
                }
            )
        else:
            convert_dict.update(
                {
                    "mlp/c_fc/weight:0": (
                        self.linear_weight_convert,
                        [".layer.2.DenseReluDense.wi.weight"],
                    ),
                    "mlp/c_proj/weight:0": (
                        self.linear_weight_convert,
                        [".layer.2.DenseReluDense.wo.weight"],
                    ),
                }
            )
        return convert_dict

    @property
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in T5."""
        convert_dict = {}
        convert_dict["wte/weight:0"] = (
            self.token_embed_weight_convert,
            ["shared.weight"],
        )
        if not cast(T5Config, self.config).tie_word_embeddings:
            convert_dict["head_fc/weight:0"] = (
                self.head_weight_convert,
                ["lm_head.weight"],
            )
        convert_dict[ENCODER_PREFIX + "/wpe/weight:0"] = (
            self.pos_embed_weight_convert,
            ["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
        )
        convert_dict[DECODER_PREFIX + "/wpe/weight:0"] = (
            self.pos_embed_weight_convert,
            ["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
        )
        convert_dict[ENCODER_PREFIX + "/ln_f/gamma:0"] = (
            self.ln_weight_convert,
            ["encoder.final_layer_norm.weight"],
        )
        convert_dict[DECODER_PREFIX + "/ln_f/gamma:0"] = (
            self._decoder_final_ln_weight_convert,
            ["decoder.final_layer_norm.weight"],
        )
        return convert_dict

    @property
    def encoder_layer_prefix(self) -> str:
        """The layer name prefix used before T5 encoder's transformer layer number."""
        return "encoder.block."

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before T5 decoder's transformer layer number."""
        return "decoder.block."

    @property
    def encoder_layer_num(self) -> int:
        """The number of transformer layers in T5 encoder."""
        return cast(T5Config, self.config).num_layers

    @property
    def encoder_hidden_size(self) -> int:
        """The hidden size of T5 encoder."""
        return cast(T5Config, self.config).d_model

    @property
    def encoder_num_attention_heads(self) -> int:
        """The number of attention heads of T5 encoder."""
        return cast(T5Config, self.config).num_heads

    @property
    def encoder_head_size(self) -> int:
        """The head size of T5 encoder."""
        return cast(T5Config, self.config).d_kv

    @property
    def decoder_layer_num(self) -> int:
        """The number of transformer layers in T5 decoder."""
        return cast(T5Config, self.config).num_decoder_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size of T5 decoder."""
        return cast(T5Config, self.config).d_model

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads of T5 decoder."""
        return cast(T5Config, self.config).num_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads of t5 decoder."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of T5 decoder."""
        return cast(T5Config, self.config).d_kv
