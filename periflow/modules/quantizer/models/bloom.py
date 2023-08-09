# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow BloomForCausalLM QuantizerHook."""

from __future__ import annotations

import torch
from transformers import PretrainedConfig
from transformers.models.bloom import BloomForCausalLM, BloomConfig  # type: ignore[import]
from typing import cast

from periflow.modules.quantizer.base import Int8QuantScaleInput, SmoothQuantHook
from periflow.modules.quantizer.formatter import ModuleName


class SmoothQuantBloomHook(SmoothQuantHook):
    """SmoothQuant Hook for BloomForCausalLM."""

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_heads = cast(BloomConfig, self.model_config).num_attention_heads
        self.hidden_size = cast(BloomConfig, self.model_config).hidden_size
        self.head_size = self.hidden_size // self.num_heads


    def get_smooth_norm_weights(self, model: BloomForCausalLM):
        """Returns iterator of layernorm's weight and linear layer's weight per transformer layer in BloomForCausalLM."""
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

    def sort_qkv_output_stats(self, max_output_stat: torch.Tensor) -> torch.Tensor:
        """Sort max_output_stas for seperating qkv_layer's output_stats"""

        split_qkv_output_stat = torch.split(max_output_stat, self.head_size)
        qkv_output_stat = [
            torch.cat(
                    [
                        split_qkv_output_stat[j * 3 + i]
                        for j in range(self.num_heads)
                    ],
                ) for i in range(3)
        ]
        qkv_output_stat = torch.cat(qkv_output_stat)
        return qkv_output_stat
    

    def get_quant_inputs(self, model: BloomForCausalLM):
        """Returns the layers which should be quantized in transformer layer of BloomForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            self_attn = decoder_layer.self_attention
            
            split_qkv_weight_list = torch.split(self_attn.query_key_value.weight, self.head_size, dim=0)
            qkv_weight_list = [
                torch.cat(
                    [
                        split_qkv_weight_list[j * 3 + i]
                        for j in range(self.num_heads)
                    ],
                    dim=0,
                ).reshape(-1, self.hidden_size)
                for i in range(3)
            ]
            qkv_weight = torch.cat(qkv_weight_list, dim=0)
            qkv_weight_out_dim = qkv_weight.size(0)

            fc1 = decoder_layer.mlp.dense_h_to_4h
            fc2 = decoder_layer.mlp.dense_4h_to_h
            yield Int8QuantScaleInput(
                layer_index=index, 
                q=(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    0,
                    qkv_weight_out_dim // 3, 
                ),
                k=(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    qkv_weight_out_dim // 3, 
                    qkv_weight_out_dim // 3 * 2, 
                ),
                v=(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    qkv_weight_out_dim // 3 * 2,
                    qkv_weight_out_dim,
                ),
                attn_fc=(
                    self_attn.dense.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.dense",
                    None,
                    None,
                ),
                ff1=(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.dense_h_to_4h",
                    None,
                    None,
                ),
                ff2=(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.dense_4h_to_h",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self):
        """Returns the linear layer types in BloomForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self):
        """Returns the prefix of the transformer layer in BloomForCausalLM."""
        return "transformer.h"
