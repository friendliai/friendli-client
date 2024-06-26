# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Int8 Quantizer Base."""

from __future__ import annotations

from typing import List, Tuple

import torch


@torch.no_grad()
def perform_smoothing(
    pre_act_params: List[torch.Tensor],
    post_act_params: List[torch.Tensor],
    activation_max: torch.Tensor,
    *,
    migration_strength: float = 0.5,
    epsilon: float = 1e-5,
    inplace: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Perform activation-weight smoothing in SmoothQuant.

    Performs the activation-weight smoothing scheme described in SmoothQuant
    (Xiao et al., 2023), which migrates the amplitude of outliers from activations
    to weights of matmul layers. The function takes in the following parameters:

    Args:
        pre_act_params: torch.Tensors representing affine parameters
            before each matmul layer.
        post_act_params: torch.Tensors representing the weight matrices of the matmul layer.
        activation_max: The maximum activation value of inputs of the matmul layer.
        migration_strength: the strength of the activation migration. Default is 0.5.
        epsilon: The epsilon used for numerical stability when calculating the scales.
            Default is 1e-5.

    Returns:
        A tuple of three torch.Tensors: (smoothed_pre_act_params, smoothed_post_act_params)

    The function calculates "scales" as `pow(|Activation|, migration_strength) /
    pow(|Weight|, 1-migration_strength)` and applies the smoothing effect into
    a normalization layer that exists before every matmul layer. This is done because
    it is more efficient than introducing a new smoothing layer before every matmul layer.
    Fusing the smoothing effect into the normalization layer results in a faster and
    more efficient implementation of the smoothing scheme.

    The function returns the smoothed normalization coefficients and the smoothed weight
    matrices after the smoothing process.
    """
    # shape of activation norms: [InChannels]
    # shape of fc weights: [OutChannels, InChannels]
    # shape of activation_max: [InChannels]

    # pylint: disable=too-many-locals
    assert pre_act_params
    assert post_act_params

    in_channels = pre_act_params[0].size(0)
    device = pre_act_params[0].device
    dtype = pre_act_params[0].dtype

    for pre_act_param in pre_act_params:
        assert pre_act_param.device == device
        assert pre_act_param.dtype == dtype

    for weight in post_act_params:
        assert weight.ndim == 2
        assert weight.size(1) == in_channels, (weight.size(), in_channels)
        assert weight.device == device

    activation_max = activation_max.to(device=device)
    weight_max = post_act_params[0].abs().max(dim=0).values
    for weight in post_act_params[1:]:
        weight_max = torch.maximum(weight_max, weight.abs().max(dim=0).values)

    assert tuple(activation_max.size()) == (in_channels,)
    assert tuple(weight_max.size()) == (in_channels,)
    alpha = migration_strength
    scales = (
        (
            activation_max.to(dtype=torch.float32).pow(alpha)
            / weight_max.to(dtype=torch.float32).pow(1 - alpha)
        )
        .clamp(min=epsilon)
        .to(dtype=dtype)
    )

    scaled_pre_act_params = [act_norm / scales for act_norm in pre_act_params]
    scaled_weights = [w * scales.view(1, -1) for w in post_act_params]

    if inplace:
        for dst, src in zip(pre_act_params, scaled_pre_act_params):
            dst.copy_(src)
        for dst, src in zip(post_act_params, scaled_weights):
            dst.copy_(src)

    return scaled_pre_act_params, scaled_weights
