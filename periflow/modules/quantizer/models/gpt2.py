# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPT2LMHeadModel QuantizerHook."""

from __future__ import annotations

from typing import Iterator, List, Tuple, Type

import torch
from transformers.models.gpt2 import GPT2LMHeadModel  # type: ignore[import]
from transformers.pytorch_utils import Conv1D  # type: ignore[import]

from periflow.modules.quantizer.base import (
    Int8QuantInput,
    SmoothQuantHook,
    TFInt8QuantInputs,
)
from periflow.modules.quantizer.schema import ModuleName


class SmoothQuantGPT2Hook(SmoothQuantHook):
    """SmoothQuant Hook for GPT2LMHeadModel."""

    def iter_smooth_norm_weights(
        self, model: GPT2LMHeadModel
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in GPT2LMHeadModel."""
        for index, decoder_layer in enumerate(model.transformer.h):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection ] gets smoothed
            yield (
                [
                    decoder_layer.ln_1.weight.data,
                    decoder_layer.ln_1.bias.data,
                ],
                [
                    decoder_layer.attn.c_attn.weight.data.transpose(
                        0, 1
                    ),  # [OutDim, InDim]
                ],
                f"{self.quantized_layer_prefix}.{index}.attn.c_attn",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.ln_2.weight.data,
                    decoder_layer.ln_2.bias.data,
                ],
                [decoder_layer.mlp.c_fc.weight.data.transpose(0, 1)],  # [OutDim, InDim]
                f"{self.quantized_layer_prefix}.{index}.mlp.c_fc",
            )

    def iter_quant_inputs(self, model: GPT2LMHeadModel) -> Iterator[TFInt8QuantInputs]:
        """Returns the layers which should be quantized in transformer block of GPT2LMHeadModel."""
        for index, decoder_layer in enumerate(model.transformer.h):
            attn = decoder_layer.attn
            attn_weight_outdim = attn.c_attn.nf  # OutDim
            fc1 = decoder_layer.mlp.c_fc
            fc2 = decoder_layer.mlp.c_proj

            yield TFInt8QuantInputs(
                layer_index=index,
                q=Int8QuantInput(
                    attn.c_attn.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}.{index}.attn.c_attn",
                    0,
                    attn_weight_outdim // 3,
                ),
                k=Int8QuantInput(
                    attn.c_attn.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}.{index}.attn.c_attn",
                    attn_weight_outdim // 3,
                    attn_weight_outdim // 3 * 2,
                ),
                v=Int8QuantInput(
                    attn.c_attn.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}.{index}.attn.c_attn",
                    attn_weight_outdim // 3 * 2,
                    attn_weight_outdim,
                ),
                attn_fc=Int8QuantInput(
                    attn.c_proj.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}.{index}.attn.c_proj",
                    None,
                    None,
                ),
                ff1=Int8QuantInput(
                    fc1.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}.{index}.mlp.c_fc",
                    None,
                    None,
                ),
                ff2=Int8QuantInput(
                    fc2.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}.{index}.mlp.c_proj",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in GPT2LMHeadModel."""
        return (Conv1D,)

    @property
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block in GPT2LMHeadModel."""
        return "transformer.h"
