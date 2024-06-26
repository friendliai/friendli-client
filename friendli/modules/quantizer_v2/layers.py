# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Quantization Layers."""

from __future__ import annotations

from typing import Optional, cast

import torch

from friendli.modules.quantizer_v2.schema.data import (
    WeightActQuantResult,
    WeightOnlyQuantResult,
)


class WeightOnlyQuantizedLinearLayer(torch.nn.Module):
    """Linear Layer with weight only quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        zeros: Optional[torch.nn.Parameter] = None,
        bias: Optional[torch.nn.Parameter] = None,
    ):
        """Initialize the Weight Only Quantized Linear Layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_scale = torch.nn.Parameter(weight_scale)
        self.weight = torch.nn.Parameter(q_weight, requires_grad=False)
        self.register_parameter("zeros", zeros)
        self.register_parameter("bias", bias)

    @staticmethod
    def from_layer(
        layer: torch.nn.Module, quant_result: WeightOnlyQuantResult
    ) -> torch.nn.Module:
        """Returns the quantized layer from the original layer."""
        zeros = (
            torch.nn.Parameter(quant_result.zero_point)
            if quant_result.zero_point
            else None
        )
        return WeightOnlyQuantizedLinearLayer(
            cast(torch.nn.Linear, layer).in_features,
            cast(torch.nn.Linear, layer).out_features,
            quant_result.q_weight,
            quant_result.weight_scale,
            zeros,
            cast(torch.nn.Linear, layer).bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization. Not used in conversion."""
        raise NotImplementedError("Not used in conversion.")


class WeightActQuantizedLinearLayer(torch.nn.Module):
    """Linear Layer with weight-act quantization."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        act_scale: torch.Tensor,
        bias: Optional[torch.nn.Parameter] = None,
    ):
        """Initialize the Weight Only Quantized Linear Layer."""
        super().__init__()
        self.in_scale = torch.nn.Parameter(act_scale)
        self.weight_scale = torch.nn.Parameter(weight_scale)
        self.weight = torch.nn.Parameter(q_weight, requires_grad=False)
        self.register_parameter("bias", bias)

    @staticmethod
    def from_layer(
        layer: torch.nn.Module, quant_result: WeightActQuantResult
    ) -> torch.nn.Module:
        """Returns the quantized layer from the original layer."""
        q_result = cast(WeightActQuantResult, quant_result)
        return WeightActQuantizedLinearLayer(
            q_result.q_weight,
            q_result.weight_scale,
            q_result.act_scale,
            cast(torch.nn.Linear, layer).bias if hasattr(layer, "bias") else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization. Not used in conversion."""
        raise NotImplementedError("Not used in conversion.")
