# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPTNeoXForCausalLM QuantizerHook."""

from __future__ import annotations

import torch
from transformers.models.gpt_neox import GPTNeoXForCausalLM  # type: ignore[import]

from periflow.modules.quantizer.base import Int8QuantScaleInput, SmoothQuantHook


class SmoothQuantGPTNeoXHook(SmoothQuantHook):
    """SmoothQuant Hook for GPTNeoXForCausalLM."""

    def get_smooth_norm_weights(self, model: GPTNeoXForCausalLM):
        """Returns iterator of layernorm's weight and linear layer's weight per transformer layer in GPTNeoXForCausalLM."""
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

    def get_quant_inputs(self, model: GPTNeoXForCausalLM):
        """Returns the layers which should be quantized in transformer layer of GPTNeoXForCausalLM."""
        for index, decoder_layer in enumerate(model.gpt_neox.layers):
            attention = decoder_layer.attention
            attention_weight_outdim = attention.query_key_value.weight.size(0)  # OutDim
            fc1 = decoder_layer.mlp.dense_h_to_4h
            fc2 = decoder_layer.mlp.dense_4h_to_h

            yield Int8QuantScaleInput(
                layer_index=index,
                q=(
                    attention.query_key_value.weight,
                    f"{self.quantized_layer_prefix}.{index}.attention.query_key_value",
                    0,
                    attention_weight_outdim // 3,
                ),
                k=(
                    attention.query_key_value.weight,
                    f"{self.quantized_layer_prefix}.{index}.attention.query_key_value",
                    attention_weight_outdim // 3,
                    attention_weight_outdim // 3 * 2,
                ),
                v=(
                    attention.query_key_value.weight,
                    f"{self.quantized_layer_prefix}.{index}.attention.query_key_value",
                    attention_weight_outdim // 3 * 2,
                    attention_weight_outdim,
                ),
                attn_fc=(
                    attention.dense.weight,
                    f"{self.quantized_layer_prefix}.{index}.attention.dense",
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
        """Returns the linear layer types in GPTNeoXForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self):
        """Returns the prefix of the transformer layer in GPTNeoXForCausalLM."""
        return "gpt_neox.layers"
