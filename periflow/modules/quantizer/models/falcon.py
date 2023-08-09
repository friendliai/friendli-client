# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow FalconForCausalLM QuantizerHook."""

from __future__ import annotations

from typing import cast

import torch
from transformers.models.falcon import (  # type: ignore[import]
    FalconConfig,
    FalconForCausalLM,
)

from periflow.modules.quantizer.base import Int8QuantScaleInput, SmoothQuantHook


class SmoothQuantFalconHook(SmoothQuantHook):
    """SmoothQuant Hook for FalconForCausalLM."""

    def get_smooth_norm_weights(self, model: FalconForCausalLM):
        """Returns iterator of layernorm's weight and linear layer's weight per transformer layer in FalconForCausalLM."""
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

    def get_quant_inputs(self, model: FalconForCausalLM):
        """Returns the layers which should be quantized in transformer layer of FalconForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            self_attn = decoder_layer.self_attention
            attn_weight_outdim = self_attn.query_key_value.weight.size(0)  # OutDim
            fc1 = decoder_layer.mlp.dense_h_to_4h
            fc2 = decoder_layer.mlp.dense_4h_to_h

            yield Int8QuantScaleInput(
                layer_index=index,
                q=(
                    self_attn.query_key_value.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    0,
                    attn_weight_outdim // 3,
                ),
                k=(
                    self_attn.query_key_value.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    attn_weight_outdim // 3,
                    attn_weight_outdim // 3 * 2,
                ),
                v=(
                    self_attn.query_key_value.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attention.query_key_value",
                    attn_weight_outdim // 3 * 2,
                    attn_weight_outdim,
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
        """Returns the linear layer types in FalconForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self):
        """Returns the prefix of the transformer layer in FalconForCausalLM."""
        return "transformer.h"
