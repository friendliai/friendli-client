# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CodeGenForCausalLM QuantizerHook."""

from __future__ import annotations

import torch
from transformers.models.codegen import CodeGenForCausalLM  # type: ignore[import]

from periflow.modules.quantizer.base import Int8QuantScaleInput, SmoothQuantHook


class SmoothQuantCodeGenHook(SmoothQuantHook):
    """SmoothQuant Hook for CodeGenForCausalLM."""

    def get_smooth_norm_weights(self, model: CodeGenForCausalLM):
        """Returns iterator of layernorm's weight and linear layer's weight per transformer layer in CodeGenForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection, MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.ln_1.weight.data,
                    decoder_layer.ln_1.bias.data,
                ],
                [
                    decoder_layer.attn.qkv_proj.weight.data,
                    decoder_layer.mlp.fc_in.weight.data,
                ],
                f"{self.quantized_layer_prefix}.{index}.attn.qkv_proj",  # the input tensors fed into qkv_proj and mlp.fc_in are identical.
            )

    def get_quant_inputs(self, model: CodeGenForCausalLM):
        """Returns the layers which should be quantized in transformer layer of CodeGenForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            self_attn = decoder_layer.attn
            attn_weight_outdim = self_attn.qkv_proj.weight.size(0)  # OutDim
            fc1 = decoder_layer.mlp.fc_in
            fc2 = decoder_layer.mlp.fc_out

            yield Int8QuantScaleInput(
                layer_index=index,
                q=(
                    self_attn.qkv_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.qkv_proj",
                    0,
                    attn_weight_outdim // 3,
                ),
                k=(
                    self_attn.qkv_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.qkv_proj",
                    attn_weight_outdim // 3,
                    attn_weight_outdim // 3 * 2,
                ),
                v=(
                    self_attn.qkv_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.qkv_proj",
                    attn_weight_outdim // 3 * 2,
                    attn_weight_outdim,
                ),
                attn_fc=(
                    self_attn.out_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.out_proj",
                    None,
                    None,
                ),
                ff1=(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.fc_in",
                    None,
                    None,
                ),
                ff2=(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.fc_out",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self):
        """Returns the linear layer types in CodeGenForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self):
        """Returns the prefix of the transformer layer in CodeGenForCausalLM."""
        return "transformer.h"
