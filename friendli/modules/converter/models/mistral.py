# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Mistral Checkpoint Converter."""


from __future__ import annotations

from typing import Any, Dict, cast

from transformers import MistralConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.models.llama import LlamaForCausalLMConverter


class MistralForCausalLMConverter(LlamaForCausalLMConverter):
    """MistralForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Mistral architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if cast(MistralConfig, self.config).hidden_act not in ["silu"]:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'hidden_act={cast(MistralConfig, self.config).hidden_act}'",
                    valid_options=["silu"],
                )
            if cast(MistralConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )

            if cast(MistralConfig, self.config).rms_norm_eps not in (1e-5, 1e-6):
                raise NotSupportedCheckpointError(
                    invalid_option=f"'rms_norm_eps={cast(MistralConfig, self.config).rms_norm_eps}'",
                    valid_options=[1e-5, 1e-6],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(MistralConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The Mistral model does not rely on "
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
            "ff_intermediate_size": self.decoder_ff_intermediate_size,
            "max_length": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
            "attention_window_size": self.attention_window_size,  # for sliding window,
            "rope_theta": self.rotary_emb_base,
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "mistral"

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in Mistral."""
        return cast(MistralConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in Mistral."""
        return cast(MistralConfig, self.config).hidden_size

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of linear layer in Mistral MLP."""
        return cast(MistralConfig, self.config).intermediate_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in Mistral."""
        return cast(MistralConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in Mistral."""
        config = cast(MistralConfig, self.config)
        if config.num_key_value_heads is None:
            return self.decoder_num_attention_heads
        return config.num_key_value_heads

    @property
    def attention_window_size(self) -> int:
        """The size of sliding window attention in Mistral."""
        return cast(MistralConfig, self.config).sliding_window
