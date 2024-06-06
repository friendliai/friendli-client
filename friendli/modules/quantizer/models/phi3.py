# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Phi3ForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Iterator, List, Tuple, Type

import torch

from friendli.modules.quantizer.base import FP8QuantHook
from friendli.modules.quantizer.schema.data import (
    HFQuantInput,
    HFTFQuantInputs,
    TFQuantInputs,
)


class Phi3Hook(FP8QuantHook):
    """FP8QuantHook for Phi3ForCausalLM."""

    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the transformer blocks in Phi3ForCausalLM."""
        return model.model.layers

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in Phi3ForCausalLM."""
        return (torch.nn.Linear,)

    def iter_tf_quant_inputs(
        self, model: torch.nn.Module
    ) -> Iterator[TFQuantInputs] | Iterator[HFTFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of Phi3ForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.self_attn
            mlp = decoder_layer.mlp

            yield HFTFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                quant_inputs=[
                    HFQuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.self_attn.qkv_proj",
                        ],
                        local_names=["qkv_proj"],
                    ),
                    HFQuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.self_attn.o_proj",
                        ],
                        local_names=[
                            "o_proj",
                        ],
                    ),
                    HFQuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.mlp.gate_up_proj",
                        ],
                        local_names=["gate_up_proj"],
                    ),
                    HFQuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.mlp.down_proj"
                        ],
                        local_names=["down_proj"],
                    ),
                ],
            )
