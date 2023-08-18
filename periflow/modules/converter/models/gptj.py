# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPTJ Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

import numpy as np
import torch
from transformers import GPTJConfig  # type: ignore[import]

from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import SUPPORTED_GELU_FAMILY, DecoderOnlyConverter
from periflow.modules.converter.interface import DECODER_PREFIX
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    get_tensor_from_state_dict,
)


class GPTJForCausalLMConverter(DecoderOnlyConverter):
    """GPTJForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if GPTJ architectures' config can be converted to PeriFlow format."""
        super().check_config()
        try:
            if (
                cast(GPTJConfig, self.config).activation_function
                not in SUPPORTED_GELU_FAMILY
            ):
                raise NotSupportedCheckpointError(
                    invalid_option="'activation_function="
                    f"{cast(GPTJConfig, self.config).activation_function}'",
                    valid_options=SUPPORTED_GELU_FAMILY,
                )
            if cast(GPTJConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if cast(GPTJConfig, self.config).layer_norm_epsilon != 1e-5:
                raise NotSupportedCheckpointError(
                    invalid_option="'layer_norm_epsilon="
                    f"{cast(GPTJConfig, self.config).layer_norm_epsilon}'",
                    valid_options=[1e-5],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def qkv_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],  # type: ignore[override]
    ) -> np.ndarray:
        """qkv_weight_convert for GPTJ's attention layer."""
        assert len(per_layer_postfixes) == 3, str(per_layer_postfixes)
        qkv_weight = torch.cat(
            [
                get_tensor_from_state_dict(state_dict, layer + postfix)
                for postfix in per_layer_postfixes
            ],
            dim=0,
        )
        qkv_weight = qkv_weight.transpose(0, 1)
        return convert_tensor_to_np_array(qkv_weight, self.data_type)

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(GPTJConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The GPTJ model does not rely on "
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
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "gpt-j"

    @property
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in GPTJ."""
        return {
            "wte/weight:0": (
                self.token_embed_weight_convert,
                ["transformer.wte.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/gamma:0": (
                self.ln_weight_convert,
                ["transformer.ln_f.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/beta:0": (
                self.ln_bias_convert,
                ["transformer.ln_f.bias"],
            ),
            "head_fc/weight:0": (
                self.head_weight_convert,
                ["lm_head.weight"],
            ),
            "head_fc/bias:0": (
                self.head_weight_convert,
                ["lm_head.bias"],
            ),
        }

    @property
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer modules in GPTJ."""
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
                self.qkv_weight_convert,
                [
                    ".attn.q_proj.weight",
                    ".attn.k_proj.weight",
                    ".attn.v_proj.weight",
                ],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".attn.out_proj.weight"],
            ),
            "mlp/c_fc/weight:0": (
                self.linear_weight_convert,
                [".mlp.fc_in.weight"],
            ),
            "mlp/c_fc/bias:0": (
                self.linear_bias_convert,
                [".mlp.fc_in.bias"],
            ),
            "mlp/c_proj/weight:0": (
                self.linear_weight_convert,
                [".mlp.fc_out.weight"],
            ),
            "mlp/c_proj/bias:0": (
                self.linear_bias_convert,
                [".mlp.fc_out.bias"],
            ),
        }

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before GPTJ's transformer module number."""
        return "transformer.h."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in GPTJ."""
        return cast(GPTJConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in GPTJ."""
        return cast(GPTJConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in GPTJ."""
        return cast(GPTJConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in gpt-j."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head siez of GPTJ."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def rotary_dim(self) -> int:
        """The rotary dim in GPTJ."""
        return cast(GPTJConfig, self.config).rotary_dim
