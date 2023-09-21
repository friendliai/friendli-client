# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Utils."""

from __future__ import annotations

import os
from dataclasses import fields
from itertools import islice
from typing import Any, Dict, Iterator, List, Tuple, Type, TypeVar, cast

import torch
from datasets import Dataset, load_dataset  # type: ignore[import]
from tqdm import tqdm
from transformers import PreTrainedTokenizer  # type: ignore[import]

from periflow.errors import CheckpointQuantizationError, InvalidConfigError
from periflow.logging import logger
from periflow.modules.quantizer.schema import (
    Int8QuantResult,
    ModuleName,
    OneOfQuantConfig,
    TFInt8QuantResults,
)


def convert_to_gpt_j_params(param: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Convert weight or bias tensor with rotary embedding to gpt-j format.

    Args:
        param (torch.Tensor): Target tensor to convert. Shape must be (num_heads, head_size, ...)
        rotary_dim (int): Degree of rotary embedding

    Returns:
        Torch tensor that heads are rotated.

    Raises:
        CheckpointQuantizationError: If arguments do not satisfy the requirements.

    """
    if param.ndim < 2:
        raise CheckpointQuantizationError(
            "Tensor dimension should be greater or equal than 2 for rotary conversion, "
            f"but got {param.ndim}"
        )

    head_size = param.shape[1]
    if rotary_dim > head_size:
        raise CheckpointQuantizationError(
            f"'rotary_dim' ({rotary_dim}) should be less or equal than 'head_size' ({head_size})"
        )

    param_rot = param[:, :rotary_dim]
    param_pass = param[:, rotary_dim:]

    origin_shape = param_rot.shape
    param_rot_1 = param_rot[:, : rotary_dim // 2]
    param_rot_2 = param_rot[:, rotary_dim // 2 :]
    param_rot = torch.stack((param_rot_1, param_rot_2), dim=2).reshape(*origin_shape)

    return torch.cat((param_rot, param_pass), dim=1)


def get_torch_data_type(data_type: str):
    """Get torch data type from Enum."""
    if data_type == "fp16":
        return torch.float16
    if data_type == "fp32":
        return torch.float32
    return torch.bfloat16


T = TypeVar("T")


def batched(it: Iterator[T], n: int) -> Iterator[List[T]]:
    """Batch an iterator into lists of size n."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def build_statistics():
    """Builds the hooks for getting the max input and output activations of a model."""
    max_input_stats = {}
    max_output_stats = {}

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

    return max_input_stats, max_output_stats, create_hook


@torch.no_grad()
def collect_max_stats(
    model: torch.nn.Module,
    dataset: Dataset,
    target_classes: Tuple[Type[torch.nn.Module], ...],
    tqdm_desc: str = "Collecting stats for Smoothing Model",
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
    max_input_stats, max_output_stats, create_hook = build_statistics()
    name_mods = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, target_classes)
    ]
    device = next(model.parameters()).device

    removables = []
    for name, module in name_mods:
        removables.append(module.register_forward_hook(create_hook(name)))
    try:
        for input_dict in tqdm(dataset, desc=tqdm_desc):
            inputs = torch.tensor(input_dict["input_ids"])
            model(inputs.to(device))
    finally:
        for removable in removables:
            removable.remove()
    return max_input_stats, max_output_stats


@torch.no_grad()
def get_int8_quant_scales(
    layer_name: str,
    input_max: torch.Tensor,
    fc_weight: torch.Tensor,
    output_max: torch.Tensor,
) -> Int8QuantResult:
    """Get the quantization scales and int8 weight for a specific layer."""
    # shape of input_max: [InChannels]
    # shape of output_max: [OutChannels]
    # shape of fc_weight: [OutChannels, InChannels]
    assert input_max.ndim == 1
    assert output_max.ndim == 1

    in_channels = input_max.size(0)
    out_channels = output_max.size(0)
    assert tuple(fc_weight.size()) == (out_channels, in_channels)

    in_scale = 127.0 / float(input_max.detach().abs().max().item())
    weight_scale = 127.0 / float(fc_weight.detach().abs().max().item())
    out_scale = 127.0 / float(output_max.detach().abs().max().item())
    int8_weight = (
        (fc_weight.detach().float() * weight_scale)
        .round()
        .clip(-128, 127)
        .to(torch.int8)
        .cpu()
    )

    return Int8QuantResult(
        layer_name,
        in_scale=in_scale,
        weight_scale=weight_scale,
        out_scale=out_scale,
        int8_weight=int8_weight,
    )


def get_encoded_dataset(
    config: OneOfQuantConfig, tokenizer: PreTrainedTokenizer
) -> Dataset:
    """Get a dataset with encoded samples for SmoothQuant calibration."""
    data_cfg = config.calibration_dataset
    data_path = data_cfg.path_or_name
    data_split = data_cfg.split

    def preprocess(example) -> Dict[str, torch.Tensor]:
        truncate_length = data_cfg.max_length * 4
        while True:
            input_ids = tokenizer(
                example[data_cfg.lookup_column_name][:truncate_length],
                return_tensors="pt",
                max_length=data_cfg.max_length * 2,
                truncation=True,
                padding=False,
            ).input_ids

            if input_ids.size(1) >= data_cfg.max_length * 2 or truncate_length >= len(
                example[data_cfg.lookup_column_name]
            ):
                input_ids = input_ids[:, : data_cfg.max_length]
                break

            truncate_length *= 2
        return {"input_ids": input_ids}

    try:
        if os.path.exists(data_path):
            dataset = load_dataset(
                data_cfg.format,
                data_files=data_path,
                split=data_split,
            )
        else:
            data_name_parts = data_path.split(":")
            if len(data_name_parts) == 1:
                dataset = load_dataset(data_path, split=data_split)
            elif len(data_name_parts) == 2:
                data_name, subset_name = data_name_parts
                dataset = load_dataset(data_name, subset_name, split=data_split)
            else:
                raise InvalidConfigError(
                    "Dataset name is in invalid format. "
                    "(valid format: '<dataset_name>' or '<dataset_name>:<subset_name>')"
                )
    except ValueError as err:
        raise CheckpointQuantizationError(f"load_dataset failed. {str(err)}") from err

    if not isinstance(dataset, Dataset):
        raise InvalidConfigError(
            "This dataset format is not supported for the calibration."
        )
    dataset = (
        dataset.shuffle(config.seed)
        .select(range(data_cfg.num_samples))
        .select_columns([data_cfg.lookup_column_name])
        .map(function=preprocess)
    )

    logger.info(
        "Smoothquant Calibration dataset(%s), data_split(%s) is successfully loaded!",
        data_path,
        data_split,
    )

    return dataset


def get_quantized_state_dict(
    quant_result_iter: Iterator[TFInt8QuantResults],
) -> Dict[str, Any]:
    """Get quantized state dict from quant results."""
    state_dict = {}
    for quant_result in quant_result_iter:
        for filed in fields(quant_result):
            layer_name = filed.name
            layer = cast(Int8QuantResult, getattr(quant_result, layer_name))
            if isinstance(layer, Int8QuantResult):
                state_dict[
                    f"{quant_result.layer_index}.{layer_name}.int8_weight"
                ] = layer.int8_weight
                state_dict[
                    f"{quant_result.layer_index}.{layer_name}.in_scale"
                ] = torch.tensor(layer.in_scale)
                state_dict[
                    f"{quant_result.layer_index}.{layer_name}.out_scale"
                ] = torch.tensor(layer.out_scale)
                state_dict[
                    f"{quant_result.layer_index}.{layer_name}.weight_scale"
                ] = torch.tensor(layer.weight_scale)

    return state_dict
