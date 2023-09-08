# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPTJForCausalLM QuantizerHook."""

from __future__ import annotations

import copy
from typing import Iterator, List, Tuple, Type, cast

import torch
from transformers.models.gptj import GPTJForCausalLM  # type: ignore[import]
from transformers.models.gptj.modeling_gptj import GPTJBlock  # type: ignore[import]

from periflow.modules.quantizer.base import (
    Int8QuantInput,
    SmoothQuantHook,
    TFInt8QuantInputs,
)
from periflow.modules.quantizer.schema import ModuleName


class SmoothQuantGPTJHook(SmoothQuantHook):
    """SmoothQuant Hook for GPTJForCausalLM."""

    def pre_smooth(self, model: torch.nn.Module) -> torch.nn.Module:
        """Pre-procedure for SmoothQuant in GPTJForCausalLM that should be called before smooth() is called."""
        for decoder_layer in cast(GPTJForCausalLM, model).transformer.h:
            decoder_layer.add_module("ln_2", copy.deepcopy(decoder_layer.ln_1))
        return model

    def iter_smooth_norm_weights(
        self, model: GPTJForCausalLM
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in GPTJForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection] gets smoothed
            yield (
                [
                    decoder_layer.ln_1.weight.data,
                    decoder_layer.ln_1.bias.data,
                ],
                [
                    decoder_layer.attn.q_proj.weight.data,  # [OutDim, InDim]
                    decoder_layer.attn.k_proj.weight.data,  # [OutDim, InDim]
                    decoder_layer.attn.v_proj.weight.data,  # [OutDim, InDim]
                ],
                f"{self.quantized_layer_prefix}.{index}.attn.q_proj",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 1] - [ MLP FF1 ] gets smoothed
            yield (
                [
                    decoder_layer.ln_2.weight.data,
                    decoder_layer.ln_2.bias.data,
                ],
                [
                    decoder_layer.mlp.fc_in.weight.data,
                ],
                f"{self.quantized_layer_prefix}.{index}.mlp.fc_in",
            )

    def iter_quant_inputs(self, model: GPTJForCausalLM) -> Iterator[TFInt8QuantInputs]:
        """Returns the layers which should be quantized in transformer block of GPTJForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            attn = decoder_layer.attn
            fc1 = decoder_layer.mlp.fc_in
            fc2 = decoder_layer.mlp.fc_out
            yield TFInt8QuantInputs(
                layer_index=index,
                q=Int8QuantInput(
                    attn.q_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.q_proj",
                    None,
                    None,
                ),
                k=Int8QuantInput(
                    attn.k_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.k_proj",
                    None,
                    None,
                ),
                v=Int8QuantInput(
                    attn.v_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.v_proj",
                    None,
                    None,
                ),
                attn_fc=Int8QuantInput(
                    attn.out_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.out_proj",
                    None,
                    None,
                ),
                ff1=Int8QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.fc_in",
                    None,
                    None,
                ),
                ff2=Int8QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.fc_out",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in GPTJForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block in GPTJForCausalLM."""
        return "transformer.h"
