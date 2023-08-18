# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPT2 Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import GPT2Config  # type: ignore[import]

from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import SUPPORTED_GELU_FAMILY, DecoderOnlyConverter
from periflow.modules.converter.interface import DECODER_PREFIX
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    get_tensor_from_state_dict,
)


class GPT2LMHeadModelConverter(DecoderOnlyConverter):
    """GPT2LMHeadModel Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if GPT2 architectures' config can be converted to PeriFlow format."""
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
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in GPT2."""
        return {
            "wte/weight:0": (
                self.token_embed_weight_convert,
                ["transformer.wte.weight"],
            ),
            DECODER_PREFIX
            + "/wpe/weight:0": (
                self.pos_embed_weight_convert,
                ["transformer.wpe.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/gamma:0": (self.ln_weight_convert, ["transformer.ln_f.weight"]),
            DECODER_PREFIX
            + "/ln_f/beta:0": (self.ln_bias_convert, ["transformer.ln_f.bias"]),
        }

    def linear_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert linear weight in GPT2, which does not need weight transpose."""
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)

    @property
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in GPT2."""
        return {
            "ln_1/gamma:0": (
                self.ln_weight_convert,
                [".ln_1.weight"],
            ),
            "ln_1/beta:0": (
                self.ln_bias_convert,
                [".ln_1.bias"],
            ),
            "attn/c_attn/weight:0": (
                self.linear_weight_convert,
                [".attn.c_attn.weight"],
            ),
            "attn/c_attn/bias:0": (
                self.linear_bias_convert,
                [".attn.c_attn.bias"],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".attn.c_proj.weight"],
            ),
            "attn/c_proj/bias:0": (
                self.linear_bias_convert,
                [".attn.c_proj.bias"],
            ),
            "ln_2/gamma:0": (
                self.ln_weight_convert,
                [".ln_2.weight"],
            ),
            "ln_2/beta:0": (
                self.ln_bias_convert,
                [".ln_2.bias"],
            ),
            "mlp/c_fc/weight:0": (
                self.linear_weight_convert,
                [".mlp.c_fc.weight"],
            ),
            "mlp/c_fc/bias:0": (
                self.linear_bias_convert,
                [".mlp.c_fc.bias"],
            ),
            "mlp/c_proj/weight:0": (
                self.linear_weight_convert,
                [".mlp.c_proj.weight"],
            ),
            "mlp/c_proj/bias:0": (
                self.linear_bias_convert,
                [".mlp.c_proj.bias"],
            ),
        }

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before GPT2's transformer layer number."""
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
    def decoder_head_size(self) -> int:
        """The head siez of GPT2."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads
