# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPTNeoXForCausalLM QuantizerHook."""

from __future__ import annotations

from typing import Iterator, List, Tuple, Type, cast

import torch
from transformers import PretrainedConfig  # type: ignore[import]
from transformers.models.gpt_neox import (  # type: ignore[import]
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from periflow.modules.quantizer.base import (
    Int8QuantInput,
    SmoothQuantHook,
    TFInt8QuantInputs,
)
from periflow.modules.quantizer.schema import ModuleName
from periflow.modules.quantizer.utils import convert_to_gpt_j_params


class SmoothQuantGPTNeoXHook(SmoothQuantHook):
    """SmoothQuant Hook for GPTNeoXForCausalLM."""

    def __init__(self, config: PretrainedConfig):
        """Initialize SmoothQuantGPTNeoXHook."""
        super().__init__(config)
        config = cast(GPTNeoXConfig, config)
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_dim = int(self.head_size * config.rotary_pct)

    def iter_smooth_norm_weights(
        self, model: GPTNeoXForCausalLM
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in GPTNeoXForCausalLM."""
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
                f"{self.quantized_layer_prefix}.{index}.attention.query_key_value",  # the input tensors fed into Q, K, V matrices are identical.
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

    def iter_quant_inputs(
        self, model: GPTNeoXForCausalLM
    ) -> Iterator[TFInt8QuantInputs]:
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

            yield TFInt8QuantInputs(
                layer_index=index,
                q=Int8QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.attention.query_key_value",
                    0,
                    attention_weight_outdim // 3,
                    self.sort_qkv_output_stats,
                ),
                k=Int8QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.attention.query_key_value",
                    attention_weight_outdim // 3,
                    attention_weight_outdim // 3 * 2,
                    self.sort_qkv_output_stats,
                ),
                v=Int8QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}.{index}.attention.query_key_value",
                    attention_weight_outdim // 3 * 2,
                    attention_weight_outdim,
                    self.sort_qkv_output_stats,
                ),
                attn_fc=Int8QuantInput(
                    attention.dense.weight,
                    f"{self.quantized_layer_prefix}.{index}.attention.dense",
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
        """Returns the linear layer types in GPTNeoXForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block in GPTNeoXForCausalLM."""
        return "gpt_neox.layers"
