# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli LlamaForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Type

import torch

from friendli.modules.quantizer.models.llama import LlamaHook
from friendli.modules.quantizer.schema.data import (
    HFQuantInput,
    HFTFQuantInputs,
    TFQuantInputs,
)


class MixtralHook(LlamaHook):
    """FP8QuantHook for MixtralForCausalLM."""

    def iter_tf_quant_inputs(
        self, model: torch.nn.Module
    ) -> Iterator[TFQuantInputs] | Iterator[HFTFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of LlamaForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.self_attn
            block_sparse_moe = decoder_layer.block_sparse_moe
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
                ],
            )
