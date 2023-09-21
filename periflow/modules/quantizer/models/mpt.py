# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow MPTForCausalLM QuantizerHook."""

from __future__ import annotations

from typing import Iterator, List, Tuple, Type

import torch

from periflow.modules.quantizer.base import (
    Int8QuantInput,
    SmoothQuantHook,
    TFInt8QuantInputs,
)
from periflow.modules.quantizer.schema import ModuleName


class SmoothQuantMPTHook(SmoothQuantHook):
    """SmoothQuant Hook for MPTForCausalLM."""

    def iter_smooth_norm_weights(
        self, model: torch.nn.Module
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in MPTForCausalLM."""
        for index, decoder_layer in enumerate(
            model.transformer.blocks  # type: ignore[union-attr, arg-type]
        ):
            # [LayerNorm 1] - [ QKV projection ] gets smoothed
            yield (
                [decoder_layer.norm_1.weight.data],
                [decoder_layer.attn.Wqkv.weight.data],
                f"{self.quantized_layer_prefix}.{index}.attn.Wqkv",
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [decoder_layer.norm_2.weight.data],
                [decoder_layer.ffn.up_proj.weight.data],  # [OutDim, InDim]
                f"{self.quantized_layer_prefix}.{index}.ffn.up_proj",
            )

    def iter_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFInt8QuantInputs]:
        """Returns the layers which should be quantized in transformer block of MPTForCausalLM."""
        for index, decoder_layer in enumerate(
            model.transformer.blocks  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.attn
            attn_weight_outdim = self_attn.Wqkv.weight.size(0)  # OutDim
            fc1 = decoder_layer.ffn.up_proj
            fc2 = decoder_layer.ffn.down_proj

            yield TFInt8QuantInputs(
                layer_index=index,
                q=Int8QuantInput(
                    self_attn.Wqkv.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.Wqkv",
                    0,
                    attn_weight_outdim // 3,
                ),
                k=Int8QuantInput(
                    self_attn.Wqkv.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.Wqkv",
                    attn_weight_outdim // 3,
                    attn_weight_outdim // 3 * 2,
                ),
                v=Int8QuantInput(
                    self_attn.Wqkv.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.Wqkv",
                    attn_weight_outdim // 3 * 2,
                    attn_weight_outdim,
                ),
                attn_fc=Int8QuantInput(
                    self_attn.out_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.attn.out_proj",
                    None,
                    None,
                ),
                ff1=Int8QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}.{index}.ffn.up_proj",
                    None,
                    None,
                ),
                ff2=Int8QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}.{index}.ffn.down_proj",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in MPTForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block in MPTForCausalLM."""
        return "transformer.blocks"
