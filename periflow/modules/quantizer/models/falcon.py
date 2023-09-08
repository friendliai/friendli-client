# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow FalconForCausalLM QuantizerHook."""

from __future__ import annotations

from typing import Iterator, List, Tuple, Type, cast

import torch
from transformers import PretrainedConfig  # type: ignore[import]
from transformers.models.falcon import (  # type: ignore[import]
    FalconConfig,
    FalconForCausalLM,
)

from periflow.modules.quantizer.base import (
    Int8QuantInput,
    SmoothQuantHook,
    TFInt8QuantInputs,
)
from periflow.modules.quantizer.schema import ModuleName
from periflow.modules.quantizer.utils import convert_to_gpt_j_params


class SmoothQuantFalconHook(SmoothQuantHook):
    """SmoothQuant Hook for FalconForCausalLM."""

    def __init__(self, config: PretrainedConfig):
        """Initialize SmoothQuantFalconHook."""
        super().__init__(config)
        config = cast(FalconConfig, config)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_dim = self.head_size
        self.num_kv_attention_heads = self.get_num_kv_attention_heads(config)

    def get_num_kv_attention_heads(self, config: FalconConfig) -> int:
        """Returns the number of key-value attention heads in FalconForCausalLM."""
        if config.new_decoder_architecture:
            if config.num_kv_heads is not None:
                return config.num_kv_heads
            return config.num_attention_heads

        if config.multi_query:
            return 1

        if config.num_kv_heads is not None:
            return config.num_kv_heads
        return config.num_attention_heads

    def iter_smooth_norm_weights(
        self, model: FalconForCausalLM
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in FalconForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):  # type: ignore[union-attr]
            if cast(FalconConfig, self.model_config).new_decoder_architecture:
                # [LayerNorm 1] - [ QKV projection ] gets smoothed
                yield (
                    [
                        decoder_layer.ln_attn.weight.data,
                        decoder_layer.ln_attn.bias.data,
                    ],
                    [
                        decoder_layer.self_attention.query_key_value.weight.data,
                    ],
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",  # the input tensors fed into Q, K, V matrices are identical.
                )
                # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
                yield (
                    [
                        decoder_layer.ln_mlp.weight.data,
                        decoder_layer.ln_mlp.bias.data,
                    ],
                    [decoder_layer.mlp.dense_h_to_4h.weight.data],  # [OutDim, InDim]
                    f"{self.quantized_layer_prefix}.{index}.mlp.dense_h_to_4h",
                )
            else:
                # [LayerNorm 1] - [ QKV projection ] gets smoothed ( MLP FF1 is not smoothed. No LayerNorm 2. )
                yield (
                    [
                        decoder_layer.input_layernorm.weight.data,
                    ],
                    [
                        decoder_layer.self_attention.query_key_value.weight.data,
                    ],
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",  # the input tensors fed into Q, K, V matrices are identical.
                )

    def reshape_qkv_weight(
        self, attn_layer: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshapes the qkv weight in FalconForCausalLM for Quantization."""
        qkv_weight = cast(torch.nn.Linear, attn_layer.query_key_value).weight
        num_queries_per_kv = self.num_attention_heads // self.num_kv_attention_heads

        qkv_weight = qkv_weight.reshape(
            self.num_kv_attention_heads,
            num_queries_per_kv + 2,
            self.head_size,
            self.hidden_size,
        )

        q_weight = qkv_weight[:, :num_queries_per_kv].reshape(
            self.num_kv_attention_heads * num_queries_per_kv,
            self.head_size,
            self.hidden_size,
        )
        k_weight = qkv_weight[:, [-2]].reshape(
            self.num_kv_attention_heads,
            self.head_size,
            self.hidden_size,
        )
        v_weight = qkv_weight[:, [-1]].reshape(
            self.num_kv_attention_heads * self.head_size,
            self.hidden_size,
        )

        q_weight = convert_to_gpt_j_params(q_weight, self.rotary_dim)
        k_weight = convert_to_gpt_j_params(k_weight, self.rotary_dim)

        q_weight = q_weight.reshape(
            self.num_kv_attention_heads * num_queries_per_kv * self.head_size,
            self.hidden_size,
        )
        k_weight = k_weight.reshape(
            self.num_kv_attention_heads * self.head_size,
            self.hidden_size,
        )

        return q_weight, k_weight, v_weight

    def sort_qkv_output_stats(self, max_output_stat: torch.Tensor) -> torch.Tensor:
        """Sort max output stats of qkv_layer in FalconForCausalLM."""
        num_queries_per_kv = self.num_attention_heads // self.num_kv_attention_heads
        qkv_output_stat = max_output_stat.reshape(
            self.num_kv_attention_heads,
            num_queries_per_kv + 2,
            self.head_size,
        )
        q_out_stats = qkv_output_stat[:, :num_queries_per_kv].reshape(
            self.num_kv_attention_heads * num_queries_per_kv,
            self.head_size,
        )
        k_out_stats = qkv_output_stat[:, [-2]].reshape(
            self.num_kv_attention_heads,
            self.head_size,
        )
        v_out_stats = qkv_output_stat[:, [-1]].reshape(
            self.num_kv_attention_heads * self.head_size,
        )
        q_out_stats = convert_to_gpt_j_params(q_out_stats, self.rotary_dim)
        k_out_stats = convert_to_gpt_j_params(k_out_stats, self.rotary_dim)
        q_out_stats = q_out_stats.reshape(
            self.num_kv_attention_heads * num_queries_per_kv * self.head_size,
        )
        k_out_stats = k_out_stats.reshape(
            self.num_kv_attention_heads * self.head_size,
        )

        return torch.cat((q_out_stats, k_out_stats, v_out_stats), dim=0)

    def iter_quant_inputs(
        self, model: FalconForCausalLM
    ) -> Iterator[TFInt8QuantInputs]:
        """Returns the layers which should be quantized in transformer block of FalconForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            self_attn = decoder_layer.self_attention
            q_weight, k_weight, v_weight = self.reshape_qkv_weight(self_attn)
            qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            fc1 = decoder_layer.mlp.dense_h_to_4h
            fc2 = decoder_layer.mlp.dense_4h_to_h

            yield TFInt8QuantInputs(
                layer_index=index,
                q=Int8QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    0,
                    q_weight.size(0),
                    self.sort_qkv_output_stats,
                ),
                k=Int8QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    q_weight.size(0),
                    q_weight.size(0) + k_weight.size(1),
                    self.sort_qkv_output_stats,
                ),
                v=Int8QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    q_weight.size(0) + k_weight.size(1),
                    qkv_weight.size(0),
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
        """Returns the linear layer types in FalconForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block in FalconForCausalLM."""
        return "transformer.h"
