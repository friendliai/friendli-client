# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPTJForCausalLM QuantizerHook."""

from __future__ import annotations

import torch
from transformers.models.gptj import GPTJForCausalLM  # type: ignore[import]

from periflow.modules.quantizer.base import Int8QuantScaleInput, SmoothQuantHook


class SmoothQuantGPTJHook(SmoothQuantHook):
    """SmoothQuant Hook for GPTJForCausalLM."""

    def get_smooth_norm_weights(self, model: GPTJForCausalLM):
        """Returns iterator of layernorm's weight and linear layer's weight per transformer layer in GPTJForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection, MLP FF 1] gets smoothed
            yield (
                [
                    decoder_layer.ln_1.weight.data,
                    decoder_layer.ln_1.bias.data,
                ],
                [
                    decoder_layer.attn.q_proj.weight.data,  # [OutDim, InDim]
                    decoder_layer.attn.k_proj.weight.data,  # [OutDim, InDim]
                    decoder_layer.attn.v_proj.weight.data,  # [OutDim, InDim]
                    decoder_layer.mlp.fc_in.weight.data,
                ],
                f"{self.quantized_layer_prefix}.{index}.attn.q_proj",  # the input tensors fed into Q, K, V, mlp_fc_in matrices are identical.
            )

    def get_quant_inputs(self, model: GPTJForCausalLM):
        """Returns the layers which should be quantized in transformer layer of GPTJForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            attn = decoder_layer.attn
            fc1 = decoder_layer.mlp.fc_in
            fc2 = decoder_layer.mlp.fc_out
            yield Int8QuantScaleInput(
                layer_index=index,
                q=(
                    attn.q_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.q_proj",
                    None,
                    None,
                ),
                k=(
                    attn.k_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.k_proj",
                    None,
                    None,
                ),
                v=(
                    attn.v_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.v_proj",
                    None,
                    None,
                ),
                attn_fc=(
                    attn.out_proj.weight,
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
        """Returns the linear layer types in GPTJForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self):
        """Returns the prefix of the transformer layer in GPTJForCausalLM."""
        return "transformer.h"
