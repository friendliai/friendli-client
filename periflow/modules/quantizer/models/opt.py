# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow OPTForCausalLM QuantizerHook."""

from __future__ import annotations

import torch
from transformers.models.opt import OPTForCausalLM  # type: ignore[import]

from periflow.modules.quantizer.base import Int8QuantScaleInput, SmoothQuantHook


class SmoothQuantOPTHook(SmoothQuantHook):
    """SmoothQuant Hook for OPTForCausalLM."""

    def get_smooth_norm_weights(self, model: OPTForCausalLM):
        """Returns iterator of layernorm's weight and linear layer's weight per transformer layer in OPTForCausalLM."""
        for index, decoder_layer in enumerate(model.model.decoder.layers):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection ] gets smoothed
            yield (
                [
                    decoder_layer.self_attn_layer_norm.weight.data,
                    decoder_layer.self_attn_layer_norm.bias.data,
                ],
                [
                    decoder_layer.self_attn.q_proj.weight.data,  # [OutDim, InDim]
                    decoder_layer.self_attn.k_proj.weight.data,
                    decoder_layer.self_attn.v_proj.weight.data,
                ],
                f"{self.quantized_layer_prefix}.{index}.self_attn.q_proj",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.final_layer_norm.weight.data,
                    decoder_layer.final_layer_norm.bias.data,
                ],
                [decoder_layer.fc1.weight.data],  # [OutDim, InDim]
                f"{self.quantized_layer_prefix}.{index}.fc1",
            )

    def get_quant_inputs(self, model: OPTForCausalLM):
        """Returns the layers which should be quantized in transformer layer of OPTForCausalLM."""
        for index, decoder_layer in enumerate(model.model.decoder.layers):
            self_attn = decoder_layer.self_attn
            fc1 = decoder_layer.fc1
            fc2 = decoder_layer.fc2
            yield Int8QuantScaleInput(
                layer_index=index,
                q=(
                    self_attn.q_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.q_proj",
                    None,
                    None,
                ),
                k=(
                    self_attn.k_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.k_proj",
                    None,
                    None,
                ),
                v=(
                    self_attn.v_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.v_proj",
                    None,
                    None,
                ),
                attn_fc=(
                    self_attn.out_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.out_proj",
                    None,
                    None,
                ),
                ff1=(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}.{index}.fc1",
                    None,
                    None,
                ),
                ff2=(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}.{index}.fc2",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self):
        """Returns the linear layer types in OPTForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self):
        """Returns the prefix of the transformer layer in OPTForCausalLM."""
        return "model.decoder.layers"
