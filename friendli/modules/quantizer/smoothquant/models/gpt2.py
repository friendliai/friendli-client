# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli GPT2LMHeadModel QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Iterator, List, Tuple, Type, cast

import torch
from transformers.models.gpt2 import GPT2LMHeadModel  # type: ignore[import]
from transformers.pytorch_utils import Conv1D  # type: ignore[import]

from friendli.modules.quantizer.schema.config import SmoothQuantConfig
from friendli.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from friendli.modules.quantizer.smoothquant.base import SmoothQuantHook


class SmoothQuantGPT2Hook(SmoothQuantHook):
    """SmoothQuant Hook for GPT2LMHeadModel."""

    def iter_smooth_norm_weights(
        self, model: GPT2LMHeadModel
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in GPT2LMHeadModel."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args
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
                f"{self.quantized_layer_prefix}{index}.attn.c_attn",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.ln_2.weight.data,
                    decoder_layer.ln_2.bias.data,
                ],
                [decoder_layer.mlp.c_fc.weight.data.transpose(0, 1)],  # [OutDim, InDim]
                f"{self.quantized_layer_prefix}{index}.mlp.c_fc",
            )
            if quant_args.attn_fc_smoothing:
                yield (
                    [decoder_layer.attn_fc_pre_smoother.scale.data.transpose(0, 1)],
                    [decoder_layer.attn.c_proj.weight.data],
                    f"{self.quantized_layer_prefix}{index}.attn.c_proj",
                )
            if quant_args.ff2_smoothing:
                yield (
                    [decoder_layer.ff2_pre_smoother.scale.data.transpose(0, 1)],
                    [decoder_layer.mlp.c_proj.weight.data],
                    f"{self.quantized_layer_prefix}{index}.mlp.c_proj",
                )

    def iter_tf_quant_inputs(self, model: GPT2LMHeadModel) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of GPT2LMHeadModel."""
        for index, decoder_layer in enumerate(model.transformer.h):
            attn = decoder_layer.attn
            attn_weight_outdim = attn.c_attn.nf  # OutDim
            fc1 = decoder_layer.mlp.c_fc
            fc2 = decoder_layer.mlp.c_proj

            yield TFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                q=QuantInput(
                    attn.c_attn.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}{index}.attn.c_attn",
                    0,
                    attn_weight_outdim // 3,
                ),
                k=QuantInput(
                    attn.c_attn.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}{index}.attn.c_attn",
                    attn_weight_outdim // 3,
                    attn_weight_outdim // 3 * 2,
                ),
                v=QuantInput(
                    attn.c_attn.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}{index}.attn.c_attn",
                    attn_weight_outdim // 3 * 2,
                    attn_weight_outdim,
                ),
                attn_fc=QuantInput(
                    attn.c_proj.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}{index}.attn.c_proj",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    fc1.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}{index}.mlp.c_fc",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    fc2.weight.transpose(0, 1),
                    f"{self.quantized_layer_prefix}{index}.mlp.c_proj",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in GPT2LMHeadModel."""
        return (Conv1D,)

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after attention in the decoder layer."""
        return decoder_layer.attn.c_proj

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after FF1 in the decoder layer."""
        return decoder_layer.mlp.c_proj

    def get_tf_blocks(self, model: GPT2LMHeadModel) -> List[torch.nn.Module]:
        """Returns the decoder layers(transformer blocks) in the model."""
        return list(model.transformer.h)
