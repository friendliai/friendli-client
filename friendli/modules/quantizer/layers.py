# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Quantization Layers."""

from __future__ import annotations

from typing import Optional, cast

import torch

from friendli.modules.quantizer.schema.data import (
    CommonQuantResult,
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
        zeros: torch.Tensor,
        bias: Optional[torch.nn.Parameter] = None,
    ):
        """Initialize the Weight Only Quantized Linear Layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_scale = torch.nn.Parameter(weight_scale)
        self.zeros = torch.nn.Parameter(zeros, requires_grad=False)
        self.weight = torch.nn.Parameter(q_weight, requires_grad=False)
        self.register_parameter("bias", bias)

    @staticmethod
    def from_layer(
        layer: torch.nn.Module, quant_result: CommonQuantResult
    ) -> torch.nn.Module:
        """Returns the quantized layer from the original layer."""
        q_result = cast(WeightOnlyQuantResult, quant_result)
        return WeightOnlyQuantizedLinearLayer(
            cast(torch.nn.Linear, layer).in_features,
            cast(torch.nn.Linear, layer).out_features,
            q_result.q_weight,
            q_result.weight_scale,
            q_result.zero_point,
            cast(torch.nn.Linear, layer).bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization. Not used in conversion."""
        raise NotImplementedError("Not used in conversion.")


class WeightActQuantizedLinearLayer(torch.nn.Module):
    """Linear Layer with weight-act quantization."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: int,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        in_scale: torch.Tensor,
        out_scale: torch.Tensor,
        bias: Optional[torch.nn.Parameter] = None,
    ):
        """Initialize the Weight Only Quantized Linear Layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_scale = torch.nn.Parameter(in_scale)
        self.out_scale = torch.nn.Parameter(out_scale)
        self.weight_scale = torch.nn.Parameter(weight_scale)
        self.weight = torch.nn.Parameter(q_weight, requires_grad=False)
        self.register_parameter("bias", bias)

    @staticmethod
    def from_layer(
        layer: torch.nn.Module, quant_result: CommonQuantResult
    ) -> torch.nn.Module:
        """Returns the quantized layer from the original layer."""
        q_result = cast(WeightActQuantResult, quant_result)
        return WeightActQuantizedLinearLayer(
            cast(torch.nn.Linear, layer).in_features,
            cast(torch.nn.Linear, layer).out_features,
            q_result.q_weight,
            q_result.weight_scale,
            q_result.in_scale,
            q_result.out_scale,
            cast(torch.nn.Linear, layer).bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fake quantization. Not used in conversion."""
        raise NotImplementedError("Not used in conversion.")
