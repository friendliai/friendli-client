# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli MPTForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Type

import torch

from friendli.modules.quantizer.base import FP8QuantHook
from friendli.modules.quantizer.schema.data import (
    HFQuantInput,
    HFTFQuantInputs,
    TFQuantInputs,
)


class MPTHook(FP8QuantHook):
    """CommonQuantHook for MPTForCausalLM."""

    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the transformer blocks in MPTForCausalLM."""
        return model.transformer.blocks

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in MPTForCausalLM."""
        return (torch.nn.Linear,)

    def iter_tf_quant_inputs(
        self, model: torch.nn.Module
    ) -> Iterator[TFQuantInputs] | Iterator[HFTFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of MPTForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.attn
            mlp = decoder_layer.ffn

            yield HFTFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                quant_inputs=[
                    HFQuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                        ],
                        local_names=["Wqkv"],
                    ),
                    HFQuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.attn.out_proj",
                        ],
                        local_names=[
                            "out_proj",
                        ],
                    ),
                    HFQuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.ffn.up_proj",
                        ],
                        local_names=["up_proj"],
                    ),
                    HFQuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.ffn.down_proj"
                        ],
                        local_names=["down_proj"],
                    ),
                ],
            )
