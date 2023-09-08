# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow BloomForCausalLM QuantizerHook."""

from __future__ import annotations

from typing import Iterator, List, Tuple, Type, cast

import torch
from transformers import PretrainedConfig  # type: ignore[import]
from transformers.models.bloom import (  # type: ignore[import]
    BloomConfig,
    BloomForCausalLM,
)

from periflow.modules.quantizer.base import (
    Int8QuantInput,
    SmoothQuantHook,
    TFInt8QuantInputs,
)
from periflow.modules.quantizer.schema import ModuleName


class SmoothQuantBloomHook(SmoothQuantHook):
    """SmoothQuant Hook for BloomForCausalLM."""

    def __init__(self, config: PretrainedConfig):
        """Initialize SmoothQuantBloomHook."""
        super().__init__(config)
        self.num_heads = cast(BloomConfig, self.model_config).num_attention_heads
        self.hidden_size = cast(BloomConfig, self.model_config).hidden_size
        self.head_size = self.hidden_size // self.num_heads

    def iter_smooth_norm_weights(
        self, model: BloomForCausalLM
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in BloomForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection ] gets smoothed
            yield (
                [
                    decoder_layer.input_layernorm.weight.data,
                    decoder_layer.input_layernorm.bias.data,
                ],
                [
                    decoder_layer.self_attention.query_key_value.weight.data,
                ],
                f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.post_attention_layernorm.weight.data,
                    decoder_layer.post_attention_layernorm.bias.data,
                ],
                [decoder_layer.mlp.dense_h_to_4h.weight.data],  # [OutDim, InDim]
                f"{self.quantized_layer_prefix}.{index}.mlp.dense_h_to_4h",
            )

    def reshape_qkv_weight(
        self, attn_layer: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshapes the qkv weight in BloomForCausalLM for Quantization."""
        qkv_layer = cast(torch.nn.Linear, attn_layer.query_key_value)
        split_qkv_weight_list = torch.split(qkv_layer.weight, self.head_size, dim=0)
        [q_weight, k_weight, v_weight] = [
            torch.cat(
                [split_qkv_weight_list[j * 3 + i] for j in range(self.num_heads)],
                dim=0,
            ).reshape(-1, self.hidden_size)
            for i in range(3)
        ]
        return q_weight, k_weight, v_weight

    def sort_qkv_output_stats(self, max_output_stat: torch.Tensor) -> torch.Tensor:
        """Sort max_output_stas for seperating qkv_layer's output_stats."""
        split_qkv_output_stat = torch.split(max_output_stat, self.head_size)
        qkv_output_stat_list = [
            torch.cat(
                [split_qkv_output_stat[j * 3 + i] for j in range(self.num_heads)],
            )
            for i in range(3)
        ]
        qkv_output_stat = torch.cat(qkv_output_stat_list)
        return qkv_output_stat

    def iter_quant_inputs(self, model: BloomForCausalLM) -> Iterator[TFInt8QuantInputs]:
        """Returns the layers which should be quantized in transformer block of BloomForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            self_attn = decoder_layer.self_attention
            q_weight, k_weight, v_weight = self.reshape_qkv_weight(self_attn)
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qkv_weight_out_dim = qkv_weight.size(0)
            fc1 = decoder_layer.mlp.dense_h_to_4h
            fc2 = decoder_layer.mlp.dense_4h_to_h

            yield TFInt8QuantInputs(
                layer_index=index,
                q=Int8QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    0,
                    qkv_weight_out_dim // 3,
                    self.sort_qkv_output_stats,
                ),
                k=Int8QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    qkv_weight_out_dim // 3,
                    qkv_weight_out_dim // 3 * 2,
                    self.sort_qkv_output_stats,
                ),
                v=Int8QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    qkv_weight_out_dim // 3 * 2,
                    qkv_weight_out_dim,
                    self.sort_qkv_output_stats,
                ),
                attn_fc=Int8QuantInput(
                    self_attn.dense.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.dense",
                    None,
                    None,
                ),
                ff1=Int8QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.dense_h_to_4h",
                    None,
                    None,
                ),
                ff2=Int8QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.dense_4h_to_h",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in BloomForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block in BloomForCausalLM."""
        return "transformer.h"
