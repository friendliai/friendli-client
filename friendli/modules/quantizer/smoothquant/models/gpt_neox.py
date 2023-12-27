# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli GPTNeoXForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple, Type, cast

import torch
from transformers.models.gpt_neox import (  # type: ignore[import]
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from friendli.modules.converter.base import OneOfConverter
from friendli.modules.converter.utils import convert_to_gpt_j_params
from friendli.modules.quantizer.schema.config import SmoothQuantConfig
from friendli.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from friendli.modules.quantizer.smoothquant.base import SmoothQuantHook


class SmoothQuantGPTNeoXHook(SmoothQuantHook):
    """SmoothQuant Hook for GPTNeoXForCausalLM."""

    def __init__(self, quant_config: Dict[str, Any], converter: OneOfConverter):
        """Initialize SmoothQuantGPTNeoXHook."""
        super().__init__(quant_config, converter)
        config = cast(GPTNeoXConfig, converter.config)
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_dim = int(self.head_size * config.rotary_pct)

    def iter_smooth_norm_weights(
        self,
        model: GPTNeoXForCausalLM,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in GPTNeoXForCausalLM."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args
        for index, decoder_layer in enumerate(model.gpt_neox.layers):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection ] gets smoothed
            yield (
                [
                    decoder_layer.input_layernorm.weight.data,
                    decoder_layer.input_layernorm.bias.data,
                ],
                [
                    decoder_layer.attention.query_key_value.weight.data,
                ],
                f"{self.quantized_layer_prefix}{index}.attention.query_key_value",  # the input tensors fed into Q, K, V matrices are identical.
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
                    [decoder_layer.attention.dense.weight.data],
                    f"{self.quantized_layer_prefix}{index}.attention.dense",
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
        """Reshape GPTNeoXForCausalLM's qkv weight for int8 quantization."""
        qkv_weight = cast(torch.nn.Linear, attn_layer).weight
        qkv_weight = qkv_weight.reshape(
            self.num_attention_heads,
            3,
            self.head_size,
            self.hidden_size,
        )

        q_weight = qkv_weight[:, 0].reshape(
            self.num_attention_heads,
            self.head_size,
            self.hidden_size,
        )
        k_weight = qkv_weight[:, 1].reshape(
            self.num_attention_heads,
            self.head_size,
            self.hidden_size,
        )
        v_weight = qkv_weight[:, 2].reshape(
            self.num_attention_heads * self.head_size,
            self.hidden_size,
        )

        q_weight = convert_to_gpt_j_params(param=q_weight, rotary_dim=self.rotary_dim)
        k_weight = convert_to_gpt_j_params(param=k_weight, rotary_dim=self.rotary_dim)
        q_weight = q_weight.reshape(
            self.num_attention_heads * self.head_size,
            self.hidden_size,
        )
        k_weight = k_weight.reshape(
            self.num_attention_heads * self.head_size,
            self.hidden_size,
        )
        return q_weight, k_weight, v_weight

    def sort_qkv_output_stats(self, max_output_stat: torch.Tensor) -> torch.Tensor:
        """Sort max output stats of qkv_layer in GPTNeoXForCausalLM."""
        max_output_stat = max_output_stat.reshape(
            self.num_attention_heads,
            3,
            self.head_size,
        )
        q_output_stat = max_output_stat[:, 0].reshape(
            self.num_attention_heads,
            self.head_size,
        )
        k_output_stat = max_output_stat[:, 1].reshape(
            self.num_attention_heads,
            self.head_size,
        )
        v_output_stat = max_output_stat[:, 2].reshape(
            self.num_attention_heads * self.head_size,
        )
        q_output_stat = convert_to_gpt_j_params(q_output_stat, self.rotary_dim)
        k_output_stat = convert_to_gpt_j_params(k_output_stat, self.rotary_dim)
        q_output_stat = q_output_stat.reshape(
            self.num_attention_heads * self.head_size,
        )
        k_output_stat = k_output_stat.reshape(
            self.num_attention_heads * self.head_size,
        )
        return torch.cat((q_output_stat, k_output_stat, v_output_stat), dim=0)

    def iter_tf_quant_inputs(
        self, model: GPTNeoXForCausalLM
    ) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of GPTNeoXForCausalLM."""
        for index, decoder_layer in enumerate(model.gpt_neox.layers):
            attention = decoder_layer.attention
            attention_weight_outdim = attention.query_key_value.weight.size(0)  # OutDim
            q_weight, k_weight, v_weight = self.reshape_qkv_weight(
                attention.query_key_value
            )
            qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            fc1 = decoder_layer.mlp.dense_h_to_4h
            fc2 = decoder_layer.mlp.dense_4h_to_h

            yield TFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                q=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.attention.query_key_value",
                    0,
                    attention_weight_outdim // 3,
                    self.sort_qkv_output_stats,
                ),
                k=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.attention.query_key_value",
                    attention_weight_outdim // 3,
                    attention_weight_outdim // 3 * 2,
                    self.sort_qkv_output_stats,
                ),
                v=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.attention.query_key_value",
                    attention_weight_outdim // 3 * 2,
                    attention_weight_outdim,
                    self.sort_qkv_output_stats,
                ),
                attn_fc=QuantInput(
                    attention.dense.weight,
                    f"{self.quantized_layer_prefix}{index}.attention.dense",
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
        """Returns the linear layer types in GPTNeoXForCausalLM."""
        return (torch.nn.Linear,)

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after attention in the decoder layer."""
        return decoder_layer.attention.dense

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after FF1 in the decoder layer."""
        return decoder_layer.mlp.dense_4h_to_h

    def get_tf_blocks(self, model: GPTNeoXForCausalLM) -> List[torch.nn.Module]:
        """Returns the decoder layers(transformer blocks) in the model."""
        return list(model.gpt_neox.layers)
