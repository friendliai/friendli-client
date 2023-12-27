# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli BloomForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple, Type, cast

import torch
from transformers.models.bloom import (  # type: ignore[import]
    BloomConfig,
    BloomForCausalLM,
)

from friendli.modules.converter.base import OneOfConverter
from friendli.modules.quantizer.schema.config import SmoothQuantConfig
from friendli.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from friendli.modules.quantizer.smoothquant.base import SmoothQuantHook


class SmoothQuantBloomHook(SmoothQuantHook):
    """SmoothQuant Hook for BloomForCausalLM."""

    def __init__(self, quant_config: Dict[str, Any], converter: OneOfConverter):
        """Initialize SmoothQuantBloomHook."""
        super().__init__(quant_config, converter)
        self.num_heads = cast(BloomConfig, converter.config).num_attention_heads
        self.hidden_size = cast(BloomConfig, converter.config).hidden_size
        self.head_size = self.hidden_size // self.num_heads

    def iter_smooth_norm_weights(
        self,
        model: BloomForCausalLM,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight pr transformer block in BloomForCausalLM."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args
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
                f"{self.quantized_layer_prefix}{index}.self_attention.query_key_value",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.post_attention_layernorm.weight.data,
                    decoder_layer.post_attention_layernorm.bias.data,
                ],
                [decoder_layer.mlp.dense_h_to_4h.weight.data],  # [OutDim, InDim]
                f"{self.quantized_layer_prefix}{index}.mlp.dense_h_to_4h",
            )
            if quant_args.attn_fc_smoothing:
                yield (
                    [decoder_layer.attn_fc_pre_smoother.scale.data],
                    [decoder_layer.self_attention.dense.weight.data],
                    f"{self.quantized_layer_prefix}{index}.self_attention.dense",
                )
            if quant_args.ff2_smoothing:
                yield (
                    [decoder_layer.ff2_pre_smoother.scale.data],
                    [decoder_layer.mlp.dense_4h_to_h.weight.data],
                    f"{self.quantized_layer_prefix}{index}.mlp.dense_4h_to_h",
                )

    def reshape_qkv_weight(
        self, attn_layer: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshapes the qkv weight in BloomForCausalLM for Quantization."""
        qkv_layer = cast(torch.nn.Linear, attn_layer.query_key_value)
        split_qkv_weight_list = torch.split(qkv_layer.weight, self.head_size, dim=0)
        num_heads = cast(BloomConfig, self.converter.config).num_attention_heads

        [q_weight, k_weight, v_weight] = [
            torch.cat(
                [split_qkv_weight_list[j * 3 + i] for j in range(num_heads)],
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

    def iter_tf_quant_inputs(self, model: BloomForCausalLM) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of BloomForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            self_attn = decoder_layer.self_attention
            q_weight, k_weight, v_weight = self.reshape_qkv_weight(self_attn)
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qkv_weight_out_dim = qkv_weight.size(0)
            fc1 = decoder_layer.mlp.dense_h_to_4h
            fc2 = decoder_layer.mlp.dense_4h_to_h

            yield TFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                q=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.self_attention.query_key_value",
                    0,
                    qkv_weight_out_dim // 3,
                    self.sort_qkv_output_stats,
                ),
                k=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.self_attention.query_key_value",
                    qkv_weight_out_dim // 3,
                    qkv_weight_out_dim // 3 * 2,
                    self.sort_qkv_output_stats,
                ),
                v=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.self_attention.query_key_value",
                    qkv_weight_out_dim // 3 * 2,
                    qkv_weight_out_dim,
                    self.sort_qkv_output_stats,
                ),
                attn_fc=QuantInput(
                    self_attn.dense.weight,
                    f"{self.quantized_layer_prefix}{index}.self_attention.dense",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.dense_h_to_4h",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.dense_4h_to_h",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in BloomForCausalLM."""
        return (torch.nn.Linear,)

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after attention in the decoder layer."""
        return decoder_layer.self_attention.dense

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after FF1 in the decoder layer."""
        return decoder_layer.mlp.dense_4h_to_h

    def get_tf_blocks(self, model: BloomForCausalLM) -> List[torch.nn.Module]:
        """Returns the decoder layers(transformer blocks) in the model."""
        return list(model.transformer.h)
