# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli FalconForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple, Type, cast

import torch
from transformers.models.falcon import (  # type: ignore[import]
    FalconConfig,
    FalconForCausalLM,
)

from friendli.modules.converter.base import OneOfConverter
from friendli.modules.converter.utils import convert_to_gpt_j_params
from friendli.modules.quantizer.schema.config import SmoothQuantConfig
from friendli.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from friendli.modules.quantizer.smoothquant.base import SmoothQuantHook


class SmoothQuantFalconHook(SmoothQuantHook):
    """SmoothQuant Hook for FalconForCausalLM."""

    def __init__(self, quant_config: Dict[str, Any], converter: OneOfConverter):
        """Initialize SmoothQuantFalconHook."""
        super().__init__(quant_config, converter)
        config = cast(FalconConfig, converter.config)
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
        self,
        model: FalconForCausalLM,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in FalconForCausalLM."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args
        for index, decoder_layer in enumerate(model.transformer.h):  # type: ignore[union-attr]
            if cast(FalconConfig, self.converter.config).new_decoder_architecture:
                # [LayerNorm 1] - [ QKV projection ] gets smoothed
                yield (
                    [
                        decoder_layer.ln_attn.weight.data,
                        decoder_layer.ln_attn.bias.data,
                    ],
                    [
                        decoder_layer.self_attention.query_key_value.weight.data,
                    ],
                    f"{self.quantized_layer_prefix}{index}.self_attention.query_key_value",  # the input tensors fed into Q, K, V matrices are identical.
                )
                # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
                yield (
                    [
                        decoder_layer.ln_mlp.weight.data,
                        decoder_layer.ln_mlp.bias.data,
                    ],
                    [decoder_layer.mlp.dense_h_to_4h.weight.data],  # [OutDim, InDim]
                    f"{self.quantized_layer_prefix}{index}.mlp.dense_h_to_4h",
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
                    f"{self.quantized_layer_prefix}{index}.self_attention.query_key_value",  # the input tensors fed into Q, K, V matrices are identical.
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

    def iter_tf_quant_inputs(self, model: FalconForCausalLM) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of FalconForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            self_attn = decoder_layer.self_attention
            q_weight, k_weight, v_weight = self.reshape_qkv_weight(self_attn)
            qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            fc1 = decoder_layer.mlp.dense_h_to_4h
            fc2 = decoder_layer.mlp.dense_4h_to_h

            yield TFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                q=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.self_attention.query_key_value",
                    0,
                    q_weight.size(0),
                    self.sort_qkv_output_stats,
                ),
                k=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.self_attention.query_key_value",
                    q_weight.size(0),
                    q_weight.size(0) + k_weight.size(0),
                    self.sort_qkv_output_stats,
                ),
                v=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.self_attention.query_key_value",
                    q_weight.size(0) + k_weight.size(0),
                    qkv_weight.size(0),
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
        """Returns the linear layer types in FalconForCausalLM."""
        return (torch.nn.Linear,)

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after attention in the decoder layer."""
        return decoder_layer.self_attention.dense

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after FF1 in the decoder layer."""
        return decoder_layer.mlp.dense_4h_to_h

    def get_tf_blocks(self, model: FalconForCausalLM) -> List[torch.nn.Module]:
        """Returns the decoder layers(transformer blocks) in the model."""
        return list(model.transformer.h)
