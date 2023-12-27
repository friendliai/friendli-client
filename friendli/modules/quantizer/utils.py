# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Quantizer Utils."""

from __future__ import annotations

import os
from contextlib import contextmanager
from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import datasets  # type: ignore[import]
import torch
from accelerate import cpu_offload_with_hook  # type: ignore
from tqdm import tqdm

from friendli.errors import InvalidConfigError, QuantizationError
from friendli.logging import logger
from friendli.modules.quantizer.schema.config import CalibrationDatasetConfig
from friendli.modules.quantizer.schema.data import (
    ModuleName,
    WeightActQuantResult,
    WeightOnlyQuantResult,
)


def scale_reshape(
    params: List[torch.Tensor],
) -> torch.Tensor:
    """Reshape scale/zero of quantized layers."""
    if len(params) == 1:
        t = params[0]
    else:
        t = torch.cat(params, dim=1)
    return t


def quantized_qkv_weight_reshape(
    params: List[torch.Tensor],
) -> torch.Tensor:
    """Reshape weight of quantized qkv layers."""
    assert len(params) == 3
    qkv_weight = torch.concat(
        params,
        dim=0,
    )  # [OutDim, InDim]

    return qkv_weight.to(torch.uint8)


def quantized_linear_weight_reshape(
    params: List[torch.Tensor],
) -> torch.Tensor:
    """Reshape weight of quantized linear layers."""
    assert len(params) == 1

    return params[0].to(torch.uint8)


def get_torch_data_type(data_type: str):
    """Get torch data type from Enum."""
    if data_type == "fp16":
        return torch.float16
    if data_type == "fp32":
        return torch.float32
    return torch.bfloat16


def safe_load_datasets(data_cfg: CalibrationDatasetConfig) -> datasets.Dataset:
    """Load dataset from calibration dataset config."""
    data_path = data_cfg.path_or_name
    data_split = data_cfg.split

    try:
        if os.path.exists(data_path):
            dataset = datasets.load_dataset(
                data_cfg.format,
                data_files=data_path,
                split=data_split,
            )
        else:
            data_name_parts = data_path.split(":")
            if len(data_name_parts) == 1:
                dataset = datasets.load_dataset(data_path, split=data_split)
            elif len(data_name_parts) == 2:
                data_name, subset_name = data_name_parts
                dataset = datasets.load_dataset(
                    data_name, subset_name, split=data_split
                )
            else:
                raise InvalidConfigError(
                    "Dataset name is in invalid format. "
                    "(valid format: '<dataset_name>' or '<dataset_name>:<subset_name>')"
                )
    except ValueError as err:
        raise QuantizationError(f"datasets.load_dataset failed. {str(err)}") from err

    if not isinstance(dataset, datasets.Dataset):
        raise InvalidConfigError(
            "This dataset format is not supported for the calibration."
        )

    return dataset


T = TypeVar("T")


def batched(it: Iterator[T], n: int) -> Iterator[List[T]]:
    """Batch an iterator into lists of size n."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def build_percentile_statistics(
    scale_percentile: float,
    symmetric: bool = True,
) -> Tuple[Callable, Callable, Callable]:
    """Builds the hooks for getting the max input and output activations of a model."""
    logger.info(
        "Building percentile statistics hooks. scale_percentile: (%s)",
        scale_percentile,
    )

    max_input_M1: Dict[str, torch.Tensor] = {}
    max_input_M2: Dict[str, torch.Tensor] = {}
    max_input_num: Dict[str, torch.Tensor] = {}
    max_output_M1: Dict[str, torch.Tensor] = {}
    max_output_M2: Dict[str, torch.Tensor] = {}
    max_output_num: Dict[str, torch.Tensor] = {}

    def create_hook(name: ModuleName):
        def update_stats(
            max_M1: Dict[str, torch.Tensor],
            max_M2: Dict[str, torch.Tensor],
            max_num: Dict[str, int],
            new_t: torch.Tensor,
        ) -> None:
            # Chan's method for computing mean and variance incrementally
            new_t = new_t.detach().reshape(-1, new_t.size(-1))
            new_numel = new_t.size(0)
            new_t_M1 = new_t.to(torch.float64).mean(dim=0)
            if symmetric:
                # it is assumed samples are always centered on zero
                # in the symmetric quantization scheme
                new_t_M1.zero_()
            new_t_M2 = ((new_t.to(torch.float64) - new_t_M1) ** 2).sum(dim=0)
            try:
                pre_numel = max_num[name]
                max_num[name] += new_numel
                delta = new_t_M1 - max_M1[name]
                max_M1[name] += delta * (new_numel / max_num[name])
                max_M2[name] += new_t_M2 + torch.pow(delta, 2) * (
                    pre_numel * new_numel / max_num[name]
                )
            except KeyError:
                max_num[name] = new_numel
                max_M1[name] = new_t_M1
                max_M2[name] = new_t_M2

        def hook(module, in_t_tup, out_t):  # pylint: disable=unused-argument
            with torch.no_grad():
                in_t = in_t_tup[0]
                update_stats(max_input_M1, max_input_M2, max_input_num, in_t)
                update_stats(max_output_M1, max_output_M2, max_output_num, out_t)

        return hook

    def finish_input_stats():
        return {
            name: torch.distributions.Normal(
                loc=max_input_M1[name],
                scale=torch.sqrt(max_input_M2[name] / max_input_num[name]).clip(
                    min=1e-7
                ),
            ).icdf(
                torch.Tensor([(scale_percentile / 100.0) * 0.5 + 0.5]).to(
                    max_input_M1[name].device
                )
            )
            for name in list(max_input_M1.keys())
        }

    def finish_output_stats():
        return {
            name: torch.distributions.Normal(
                loc=max_output_M1[name],
                scale=torch.sqrt(max_output_M2[name] / max_output_num[name]).clip(
                    min=1e-7
                ),
            ).icdf(
                torch.Tensor([(scale_percentile / 100.0) * 0.5 + 0.5]).to(
                    max_output_M1[name].device
                )
            )
            for name in list(max_output_M1.keys())
        }

    return finish_input_stats, finish_output_stats, create_hook


def build_max_statistics() -> Tuple[Callable, Callable, Callable]:
    """Builds the hooks for getting the max input and output activations of a model."""
    logger.info("Building max statistics hooks")
    max_input_stats: Dict[str, torch.Tensor] = {}
    max_output_stats: Dict[str, torch.Tensor] = {}

    def create_hook(name: ModuleName):
        def hook(modules, in_t_tup, out_t):  # pylint: disable=unused-argument
            in_t = in_t_tup[0]
            in_t = (
                in_t.detach().abs().reshape(-1, in_t.size(-1)).max(dim=0).values
            )  # reduce-max only leaving the hidden dim (supposing the last dim is the hidden dim)
            out_t = out_t.detach().reshape(-1, out_t.size(-1))
            out_t = out_t.abs().max(dim=0).values
            try:
                max_input_stats[name] = torch.maximum(max_input_stats[name], in_t)
            except KeyError:
                max_input_stats[name] = in_t
            try:
                max_output_stats[name] = torch.maximum(max_output_stats[name], out_t)
            except KeyError:
                max_output_stats[name] = out_t

        return hook

    def finish_input_stats():
        return max_input_stats

    def finish_output_stats():
        return max_output_stats

    return finish_input_stats, finish_output_stats, create_hook


@torch.no_grad()
def collect_stats(
    model: torch.nn.Module,
    device: str,
    dataset: datasets.Dataset,
    target_classes: Tuple[Type[torch.nn.Module], ...],
    tqdm_desc: str,
    percentile: float,
) -> Tuple[Dict[ModuleName, torch.Tensor], Dict[ModuleName, torch.Tensor]]:
    """Collects the maximum values of input and output activations of a specific model.

    Args:
        model (torch.nn.Module): The model for which we want to collect the max statistics.
        dataset (Dataset): Dataset that contains input tensors.
        target_classes (Tuple[Type[torch.nn.Module], ...]): A tuple of the target classes.

    Returns:
        A tuple of two dictionaries: (max_input_stats, max_output_stats), where:
        max_input_stats: The maximum input activation values for each module of the model.
        max_output_stats: The maximum output activation values for each module of the model.

    This function uses a forward hook to capture the maximum input and output activation values
    of the specified target_classes. The max_batch_size parameter controls the size of the input
    batches that are passed through the model.

    The function returns two dictionaries containing the maximum input and output activation
    values for each module of the model, respectively. These dictionaries can be used to calculate
    scaling factors for weight quantization and activation smoothing.

    """
    # pylint: disable=too-many-locals
    max_input_stats, max_output_stats, create_hook = (
        build_percentile_statistics(percentile)
        if percentile < 100.0
        else build_max_statistics()
    )
    name_mods = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, target_classes)
    ]

    removables = []
    for name, module in name_mods:
        removables.append(module.register_forward_hook(create_hook(name)))
    try:
        for inp_dict in tqdm(dataset, desc=tqdm_desc):
            inputs = torch.tensor(inp_dict["input_ids"])
            model(inputs.to(device))
    finally:
        for removable in removables:
            removable.remove()
    return max_input_stats(), max_output_stats()


def build_inps_hook():
    """Builds the hooks for getting the input and output activations of a module."""
    args_dict = {}
    kwargs_dict = {}

    def create_hook(name: ModuleName):
        def hook(m, args, kwargs, y):  # pylint: disable=unused-argument
            assert name not in args_dict
            assert name not in kwargs_dict
            # assumption: all positional arguments are torch.Tensor
            args_dict[name] = [t.detach() for t in args]
            kwargs_dict[name] = {
                k: (v.detach() if isinstance(v, torch.Tensor) else v)
                for k, v in kwargs.items()
            }

        return hook

    return args_dict, kwargs_dict, create_hook


def collect_inps(
    module: torch.nn.Module,
    module_args: Tuple[Any, ...],
    module_kwargs: Dict[str, Any],
    device: str,
    target_classes: Tuple[Type[torch.nn.Module], ...],
) -> Tuple[Dict[ModuleName, Tuple[Any]], Dict[ModuleName, Dict[str, Any]]]:
    """Collects concated input and output activations of a specific module."""
    args_dict, kwargs_dict, create_hook = build_inps_hook()
    name_mods = [
        (name, m) for name, m in module.named_modules() if isinstance(m, target_classes)
    ]

    removables = []
    for name, m in name_mods:
        removables.append(m.register_forward_hook(create_hook(name), with_kwargs=True))

    module(
        *((t.to(device) if isinstance(t, torch.Tensor) else t) for t in module_args),
        **{
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in module_kwargs.items()
        },
    )

    for removable in removables:
        removable.remove()

    return args_dict, kwargs_dict


def get_torch_quant_dtype(q_bit: int = 8):
    """Get torch quant data type from quant bit."""
    if q_bit == 8:
        return torch.int8
    if q_bit == 4:
        return torch.int32  # In AWQ, we use int32 to represent int4
    raise ValueError(f"Invalid quant bit: {q_bit}")


@torch.no_grad()
def get_weight_act_quant_scales(
    layer_name: str,
    input_max: torch.Tensor,
    fc_weight: torch.Tensor,
    output_max: torch.Tensor,
    device: str = "cpu",
) -> WeightActQuantResult:
    """Get the quantization scales and int8 weight for a specific layer."""
    # shape of input_max: [InChannels]
    # shape of output_max: [OutChannels]
    # shape of fc_weight: [OutChannels, InChannels]
    assert input_max.ndim == 1
    assert output_max.ndim == 1

    in_channels = input_max.size(0)
    out_channels = output_max.size(0)
    assert tuple(fc_weight.size()) == (out_channels, in_channels)

    max_int = 2 ** (8 - 1) - 1
    min_int = -(2 ** (8 - 1))

    in_scale = float(max_int) / float(input_max.detach().abs().max().item())
    weight_scale = float(max_int) / float(fc_weight.detach().abs().max().item())
    out_scale = float(max_int) / float(output_max.detach().abs().max().item())
    q_weight = (
        (fc_weight.detach().float() * weight_scale)
        .round()
        .clip(min_int, max_int)
        .to(get_torch_quant_dtype(8))
        .to(device)
    )

    return WeightActQuantResult(
        layer_name,
        q_bit=8,
        zero_point=torch.tensor(0.0),
        in_scale=torch.tensor(in_scale),
        weight_scale=torch.tensor(weight_scale),
        out_scale=torch.tensor(out_scale),
        q_weight=q_weight,
        q_group_size=-1,
    )


def get_weight_only_quant_scales(
    w: torch.Tensor,
    q_bit: int,
    q_group_size: int,
    layer_name: str = "",
    device: Union[str, torch.device] = "cpu",
) -> WeightOnlyQuantResult:
    """Return the quantization scales of weight for a specific layer."""
    org_w_shape = w.shape  # [OutDim, InDim]

    w = w.reshape(-1, q_group_size)  # [OutDim x num_groups, group_size]
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)

    max_int = 2**q_bit - 1
    min_int = 0

    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0

    q_weight = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    q_weight = q_weight.reshape(org_w_shape).detach().to(device)
    scales = (
        scales.view(org_w_shape[0], -1).transpose(0, 1).detach().to(device)
    )  # [OutDim, num_groups]
    zeros = (
        zeros.view(org_w_shape[0], -1).transpose(0, 1).detach().to(device)
    )  # [OutDim, num_groups]

    assert torch.isnan(q_weight).sum() == 0
    return WeightOnlyQuantResult(
        layer_name,
        q_bit=q_bit,
        zero_point=zeros,
        q_group_size=q_group_size,
        weight_scale=scales,
        q_weight=q_weight,
    )


def send_model_to_device(
    model: torch.nn.Module,
    device: Union[str, torch.device],
    *,
    exclude: Iterable[torch.nn.Module] = (),
):
    """Send the model and its submodules onto device except for modules designated by `exclude`."""
    exclude_set = set(exclude)

    @torch.no_grad()
    def recurse(m: torch.nn.Module):
        if m in exclude_set:
            return
        for name, p in list(m.named_parameters(recurse=False)):
            m.register_parameter(name, torch.nn.Parameter(p.to(device)))
        for name, b in list(m.named_buffers(recurse=False)):
            m.register_buffer(name, b.to(device))

        for child in m.children():
            recurse(child)

    recurse(model)


class RemovableOffloaderHook(Protocol):
    """Hook protocol for cpu offloader."""

    def offload(self) -> None:
        """Offload the associated block onto CPU."""

    def remove(self) -> None:
        """Remove this hook."""


@contextmanager
def offload_module_sequence(
    blocks: Sequence[torch.nn.Module], device: Union[str, torch.device]
):
    """Offload a sequence of torch modules automatically.

    In the beginning, all blocks are supposed to reside on CPU.
    When i-th block is called, it is loaded onto `device` on the fly.
    And at the same time, it offloads (i-1)-th block back to CPU.
    """
    module_hooks: List[RemovableOffloaderHook] = []
    if blocks:
        prev_module_hook = None
        for tf_block in blocks:
            _, module_hook = cpu_offload_with_hook(
                tf_block, device, prev_module_hook=prev_module_hook
            )
            prev_module_hook = module_hook
            module_hooks.append(module_hook)
    try:
        yield
    finally:
        for hook in module_hooks:
            hook.offload()
        for hook in module_hooks:
            hook.remove()
