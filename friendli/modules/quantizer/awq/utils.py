# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# Copyright (c) 2023 MIT HAN Lab
# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Friendli AWQ Quantizer Util."""

from __future__ import annotations

import gc
from typing import Any, Dict, Iterable, List, Tuple

import torch


def pseudo_quantize_tensor(w: torch.Tensor, q_bit: int = 8, q_group_size: int = -1):
    """Pseudo quantize tensor."""
    org_w_shape = w.shape
    w = w.reshape(-1, q_group_size)
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**q_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    return w


@torch.no_grad()
def get_weight_scale(weight: torch.Tensor, q_group_size=-1):
    """Get weight scale for AWQ."""
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    """Get activation scale for AWQ."""
    return x.abs().view(-1, x.shape[-1]).mean(0)


def search_module_scale(
    module: torch.nn.Module,
    module_args: Tuple[Any, ...],
    module_kwargs: Dict[str, Any],
    linears2scale: Iterable[torch.nn.Linear],
    linear_inp: torch.Tensor,
    q_group_size: int,
    q_bit: int,
) -> torch.Tensor:
    """Search the AWQ scale for a module."""
    # pylint: disable=too-many-locals
    weight = torch.cat([_m.weight for _m in linears2scale], dim=0)  # type: ignore
    with torch.no_grad():
        org_out = module(*module_args, **module_kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]

    x_max = get_act_scale(linear_inp)
    w_max = get_weight_scale(weight, q_group_size)
    del weight
    gc.collect()  # type: ignore
    torch.cuda.empty_cache()

    best_error = float("inf")
    best_scales = torch.zeros(x_max.shape[0], device=x_max.device)
    n_grid = 20
    history = []
    org_sd = {k: v.to("cpu", copy=True) for k, v in module.state_dict().items()}
    for grid in range(n_grid):
        ratio = grid * 1.0 / n_grid
        scales = (x_max.pow(ratio) / w_max.pow(1 - ratio)).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for fc in linears2scale:
            fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))  # type: ignore
            fc.weight.data = pseudo_quantize_tensor(
                w=fc.weight.data,  # type: ignore
                q_bit=q_bit,
                q_group_size=q_group_size,
            ) / (scales.view(1, -1))

        out = module(*module_args, **module_kwargs)
        if isinstance(out, tuple):
            out = out[0]

        loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_scales = scales
        module.load_state_dict(org_sd)
    best_scales = best_scales.view(-1)

    assert torch.isnan(best_scales).sum() == 0, best_scales
    return best_scales.detach()


def apply_module_scale(
    prev_ops: List[torch.nn.Module],
    linear_layers: Iterable[torch.nn.Linear],
    scales: torch.Tensor,
) -> None:
    """Apply AWQ Scale for Module, and return the scaled input for Clipping."""
    for prev_op in prev_ops:
        for _, param in prev_op.named_parameters(recurse=False):
            if isinstance(prev_op, torch.nn.Linear):
                # TODO: handle bias
                assert len(param.data.shape) == 2
                param.data.div_(scales.view(-1, 1))
            else:
                assert param.data.shape == scales.shape
                param.data.div_(scales)

    for layer in linear_layers:
        layer.weight.data.mul_(scales.view(1, -1))


def search_module_clip(
    w: torch.Tensor,
    inp: torch.Tensor,
    q_group_size: int,
    q_bit: int,
    n_grid=20,
    max_shrink=0.5,
    n_sample_token=512,
) -> torch.Tensor:
    """Search the best clip for a module."""
    # pylint: disable=too-many-locals
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # inp  [n_token, ci] -> [1, n_token, n_group, group size]
    w = w.view(w.shape[0], 1, -1, q_group_size)

    inp = inp.view(-1, inp.shape[-1])
    inp = inp.reshape(1, inp.shape[0], -1, q_group_size)
    inp = inp[:, 0 :: inp.shape[1] // n_sample_token]

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        inp = inp.to(w.device)
        org_out = (inp * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(
                w=cur_w,
                q_bit=q_bit,
                q_group_size=q_group_size,
            )
            cur_out = (inp * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del inp
    del org_out
    gc.collect()
    torch.cuda.empty_cache()

    return best_max_val.squeeze(1)


def apply_module_clip(
    max_val: torch.Tensor,
    layer: torch.nn.Linear,
):
    """Apply AWQ Clip for Module."""
    max_val = max_val.to(layer.weight.device)  # type: ignore
    org_shape = layer.weight.shape
    layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)  # type: ignore
    layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
    layer.weight.data = layer.weight.data.reshape(org_shape)  # type: ignore
