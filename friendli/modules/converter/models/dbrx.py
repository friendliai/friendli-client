# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Dbrx Checkpoint Converter."""


from __future__ import annotations

from typing import cast

from transformers.models.dbrx.configuration_dbrx import (  # type: ignore[import]
    DbrxConfig,
    DbrxFFNConfig,
)

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.modules.converter.base import FP8OnlyConverter
from friendli.modules.converter.interface import RotaryEmbeddingConversionInterface


class DbrxForCausalLMConverter(FP8OnlyConverter, RotaryEmbeddingConversionInterface):
    """DbrxForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Dbrx architectures' config can be converted to Friendli format."""
        super().check_config()
        config = cast(DbrxConfig, self.config)
        try:
            if config.tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if config.ffn_config.moe_top_k not in [1, 2, 4]:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'moe_top_k={config.ffn_config.moe_top_k}'",
                    valid_options=[1, 2, 4],
                )
            if config.ffn_config.moe_num_experts not in [1, 2, 4, 8, 16]:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'moe_num_experts={config.ffn_config.moe_num_experts}'",
                    valid_options=[1, 2, 4, 8, 16],
                )

        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    @property
    def model_type(self) -> str:
        """Model type."""
        return "dbrx"

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before LLaMA's transformer block number."""
        return "transformer.blocks."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in LLaMA."""
        return cast(DbrxConfig, self.config).n_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in LLaMA."""
        return cast(DbrxConfig, self.config).d_model

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in LLaMA."""
        return cast(DbrxConfig, self.config).n_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in LLaMA."""
        config = cast(DbrxConfig, self.config)
        if config.attn_config.kv_n_heads is None:
            return self.decoder_num_attention_heads
        return config.attn_config.kv_n_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of LLaMA."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in LLaMA MLP."""
        return cast(DbrxConfig, self.config).ffn_config.ffn_hidden_size

    @property
    def rotary_dim(self) -> int:
        """The rotary embedding dimension of LLaMA."""
        return self.decoder_head_size

    @property
    def rotary_emb_base(self) -> float:
        """The rotary embedding base of LLaMA."""
        return cast(DbrxConfig, self.config).attn_config.rope_theta
