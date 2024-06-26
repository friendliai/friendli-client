# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli ArcticForCausalLM QuantizerHook."""

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


class ArcticHook(FP8QuantHook):
    """FP8QuantHook for ArcticForCausalLM."""

    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the transformer blocks in ArcticForCausalLM."""
        return model.model.layers

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in ArcticForCausalLM."""
        return (torch.nn.Linear,)

    def iter_tf_quant_inputs(
        self, model: torch.nn.Module
    ) -> Iterator[TFQuantInputs] | Iterator[HFTFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of ArcticForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.self_attn
            block_sparse_moe = decoder_layer.block_sparse_moe
            mlp = decoder_layer.residual_mlp
            moe_ff1_ff_gate_target_names = []
            for expert_idx in range(self.converter.num_experts):
                moe_ff1_ff_gate_target_names.extend(
                    [
                        f"{self.quantized_layer_prefix}{index}.block_sparse_moe.experts.{expert_idx}.w1",
                        f"{self.quantized_layer_prefix}{index}.block_sparse_moe.experts.{expert_idx}.w3",
                    ]
                )

            yield HFTFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                quant_inputs=[
                    HFQuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.self_attn.q_proj",
                            f"{self.quantized_layer_prefix}{index}.self_attn.k_proj",
                            f"{self.quantized_layer_prefix}{index}.self_attn.v_proj",
                        ],
                        local_names=["q_proj", "k_proj", "v_proj"],
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
                    # router
                    HFQuantInput(
                        parent_module=block_sparse_moe,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.block_sparse_moe.gate",
                        ],
                        local_names=["gate"],
                    ),
                    # ff1, ff_gate in each moe
                    HFQuantInput(
                        parent_module=block_sparse_moe.experts,
                        target_names=moe_ff1_ff_gate_target_names,
                        local_names=["w1", "w3"],
                    ),
                    # ff2 in each moe
                    HFQuantInput(
                        parent_module=block_sparse_moe.experts,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.block_sparse_moe.experts.{expert_idx}.w2"
                            for expert_idx in range(self.converter.num_experts)
                        ],
                        local_names=["w2"],
                    ),
                    # ff1, ff_gate in parallel mlp
                    HFQuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.residual_mlp.w1",
                            f"{self.quantized_layer_prefix}{index}.residual_mlp.w3",
                        ],
                        local_names=["w1", "w3"],
                    ),
                    # ff2 in parallel mlp
                    HFQuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.residual_mlp.w2"
                        ],
                        local_names=["w2"],
                    ),
                ],
            )
