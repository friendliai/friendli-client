# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli DbrxForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple, Type, cast

import torch
from torch.nn.modules import Module
from tqdm import tqdm
from transformers.models.dbrx.modeling_dbrx import DbrxBlock, DbrxConfig, DbrxExpertGLU

from friendli.modules.quantizer.base import FP8QuantHook
from friendli.modules.quantizer.schema.data import (
    HFQuantInput,
    HFTFQuantInputs,
    TFQuantInputs,
)


class DbrxLinearLayer(torch.nn.Module):
    """Custom FF2Proj layer for DbrxForCausalLM."""

    def __init__(self, weight: torch.nn.Parameter):
        """Initialize the DbrxLinearLayer."""
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor, chunked_weight: torch.Tensor) -> torch.Tensor:
        """Forward pass for the DbrxLinearLayer."""
        return x.matmul(chunked_weight)


class CustomDbrxExpertGLU(DbrxExpertGLU):
    """Custom DbrxExpertGLU layer for DbrxForCausalLM.

    This layer is used to replace the DbrxExpertGLU layer in DbrxForCausalLM.
    For collecting input of the ff2 layer in each experts, we need to override the forward method.
    """

    def __init__(self, layer: DbrxExpertGLU, ffn_act_fn: Dict):
        """Initialize the CustomDbrxExpertGLU."""
        super().__init__(
            layer.hidden_size, layer.ffn_hidden_size, layer.moe_num_experts, ffn_act_fn
        )

        self.v1_linear = DbrxLinearLayer(layer.v1.detach())
        self.w1_linear = DbrxLinearLayer(layer.w1.detach())
        self.w2_linear = DbrxLinearLayer(layer.w2.detach())

    def forward(
        self,
        x: torch.Tensor,
        expert_w1: torch.Tensor,
        expert_v1: torch.Tensor,
        expert_w2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the CustomDbrxExpertGLU."""
        gate_proj = self.w1_linear(x, expert_w1.t())
        up_proj = self.v1_linear(x, expert_v1.t())
        gate_proj = self.activation_fn(gate_proj)
        intermediate_states = gate_proj * up_proj
        down_proj = self.w2_linear(intermediate_states, expert_w2)
        return down_proj

    @staticmethod
    def from_layer(layer: DbrxExpertGLU, config: DbrxConfig) -> CustomDbrxExpertGLU:
        """Creates a CustomDbrxExpertGLU layer from a DbrxExpertGLU layer."""
        custom_layer = CustomDbrxExpertGLU(layer, config.ffn_config.ffn_act_fn)
        custom_layer.v1 = layer.v1
        custom_layer.w1 = layer.w1
        custom_layer.w2 = layer.w2
        return custom_layer


class DbrxHook(FP8QuantHook):
    """FP8QuantHook for DbrxForCausalLM."""

    def get_quantized_param_names(self, model: torch.nn.Module) -> List[str]:
        """Return the parameter names of quantized layers."""
        quantized_param_names = []
        for index in range(
            len(self.get_tf_blocks(model))  # type: ignore[union-attr, arg-type]
        ):
            quantized_param_names.extend(
                [
                    f"{self.quantized_layer_prefix}{index}.norm_attn_norm.attn.Wqkv.weight",
                    f"{self.quantized_layer_prefix}{index}.norm_attn_norm.attn.out_proj.weight",
                    f"{self.quantized_layer_prefix}{index}.ffn.router.layer.weight",
                    f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.v1",
                    f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.w1",
                    f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.w2",
                ]
            )
        return quantized_param_names

    def get_quantized_param_scale_names(self, model: torch.nn.Module) -> List[str]:
        """Return the parameter scale names of quantized layers."""
        quantized_param_scale_names = []
        for index in range(
            len(self.get_tf_blocks(model))  # type: ignore[union-attr, arg-type]
        ):
            quantized_param_scale_names.extend(
                [
                    f"{self.quantized_layer_prefix}{index}.norm_attn_norm.attn.Wqkv.weight_scale",
                    f"{self.quantized_layer_prefix}{index}.norm_attn_norm.attn.out_proj.weight_scale",
                    f"{self.quantized_layer_prefix}{index}.ffn.router.layer.weight_scale",
                    f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.v1_weight_scale",
                    f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.w1_weight_scale",
                    f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.w2_weight_scale",
                ]
            )
            quantized_param_scale_names.extend(
                [
                    f"{self.quantized_layer_prefix}{index}.norm_attn_norm.attn.Wqkv.in_scale",
                    f"{self.quantized_layer_prefix}{index}.norm_attn_norm.attn.out_proj.in_scale",
                    f"{self.quantized_layer_prefix}{index}.ffn.router.layer.in_scale",
                    f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.v1_in_scale",
                    f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.w1_in_scale",
                    f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.w2_in_scale",
                ]
            )
        return quantized_param_scale_names

    def pre_quantize(self, model: Module) -> torch.nn.Module:
        """Pre-quantization hook for DbrxForCausalLM."""
        for decoder_layer in tqdm(
            self.get_tf_blocks(model),
            desc="Pre-quantizing DbrxForCausalLM",
            unit="layer",
        ):
            cast(
                DbrxBlock, decoder_layer
            ).ffn.experts.mlp = CustomDbrxExpertGLU.from_layer(
                cast(DbrxBlock, decoder_layer).ffn.experts.mlp, self.converter.config
            )
        return model

    def post_quantize(self, model: Module) -> torch.nn.Module:
        """Post-quantization hook for DbrxForCausalLM."""
        for decoder_layer in tqdm(
            self.get_tf_blocks(model),
            desc="Post-quantizing DbrxForCausalLM",
            unit="layer",
        ):
            mlp = cast(DbrxBlock, decoder_layer).ffn.experts.mlp

            # ff1
            setattr(mlp, "v1_in_scale", mlp.v1_linear.in_scale)
            setattr(mlp, "v1_weight_scale", mlp.v1_linear.weight_scale)
            mlp.v1 = mlp.v1_linear.weight
            del mlp.v1_linear

            # ff_gate
            setattr(mlp, "w1_in_scale", mlp.w1_linear.in_scale)
            setattr(mlp, "w1_weight_scale", mlp.w1_linear.weight_scale)
            mlp.w1 = mlp.w1_linear.weight
            del mlp.w1_linear

            # ff2
            setattr(mlp, "w2_in_scale", mlp.w2_linear.in_scale)
            setattr(mlp, "w2_weight_scale", mlp.w2_linear.weight_scale)
            mlp.w2 = mlp.w2_linear.weight
            del mlp.w2_linear
        return model

    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the transformer blocks in DbrxForCausalLM."""
        return model.transformer.blocks

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in DbrxForCausalLM."""
        return (
            torch.nn.Linear,
            DbrxLinearLayer,
        )

    def iter_tf_quant_inputs(
        self, model: torch.nn.Module
    ) -> Iterator[TFQuantInputs] | Iterator[HFTFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of DbrxForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = cast(DbrxBlock, decoder_layer).norm_attn_norm.attn
            mlp = cast(DbrxBlock, decoder_layer).ffn.experts.mlp

            yield HFTFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                quant_inputs=[
                    HFQuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.norm_attn_norm.attn.Wqkv",
                        ],
                        local_names=["Wqkv"],
                    ),
                    HFQuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.norm_attn_norm.attn.out_proj",
                        ],
                        local_names=[
                            "out_proj",
                        ],
                    ),
                    HFQuantInput(
                        parent_module=cast(DbrxBlock, decoder_layer).ffn.router,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.ffn.router.layer",
                        ],
                        local_names=["layer"],
                    ),
                    HFQuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.w1_linear",
                            f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.v1_linear",
                        ],
                        local_names=["w1_linear", "v1_linear"],
                    ),
                    HFQuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.ffn.experts.mlp.w2_linear"
                        ],
                        local_names=["w2_linear"],
                    ),
                ],
            )
