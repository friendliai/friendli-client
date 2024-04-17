# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Cohere Checkpoint Converter."""


from __future__ import annotations

from typing import cast

from transformers import CohereConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.modules.converter.base import FP8OnlyConverter
from friendli.modules.converter.interface import RotaryEmbeddingConversionInterface
from friendli.modules.converter.schema import ConvertInfo


class CohereForCausalLMConverter(FP8OnlyConverter, RotaryEmbeddingConversionInterface):
    """CohereForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if LLaMA architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if cast(CohereConfig, self.config).hidden_act not in ["silu"]:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'hidden_act={cast(CohereConfig, self.config).hidden_act}'",
                    valid_options=["silu"],
                )
            if not cast(CohereConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=False'",
                    valid_options=[True],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    @property
    def model_type(self) -> str:
        """Model type."""
        return "cohere"

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before LLaMA's transformer block number."""
        return "model.layers."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in LLaMA."""
        return cast(CohereConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in LLaMA."""
        return cast(CohereConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in LLaMA."""
        return cast(CohereConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in LLaMA."""
        config = cast(CohereConfig, self.config)
        if config.num_key_value_heads is None:
            return self.decoder_num_attention_heads
        return config.num_key_value_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of LLaMA."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in LLaMA MLP."""
        return self.config.intermediate_size

    @property
    def rotary_dim(self) -> int:
        """The rotary embedding dimension of LLaMA."""
        return self.decoder_head_size

    @property
    def rotary_emb_base(self) -> float:
        """The rotary embedding base of LLaMA."""
        return cast(CohereConfig, self.config).rope_theta
