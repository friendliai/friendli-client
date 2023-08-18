# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Utils."""

from __future__ import annotations

import os
from dataclasses import fields
from itertools import islice
from typing import Any, Dict, Iterator, List, Tuple, Type, TypeVar, cast

import torch
from datasets import load_dataset  # type: ignore[import]
from tqdm import tqdm

from periflow.errors import CheckpointQuantizationError
from periflow.logging import logger
from periflow.modules.quantizer.formatter import Int8QuantScale, Int8QuantScaleResult

ModuleName = str
QUANT_PREFIX = "quant_tf_layer"


def get_torch_data_type(data_type: str):
    """Get torch data type from Enum."""
    if data_type == "fp16":
        return torch.float16
    elif data_type == "fp32":
        return torch.float32
    else:
        return torch.bfloat16


def batched(iterable: Iterator[T], n: int) -> Iterator[List[T]]:
    """Batch an iterator into lists of size n."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
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
        def hook(module, in_t_tup, out_t):
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


T = TypeVar("T")


@torch.no_grad()
def collect_max_stats(
    model: torch.nn.Module,
    inputs: Iterator[torch.Tensor],
    target_classes: Tuple[Type[torch.nn.Module], ...],
    max_batch_size: int = 1,
) -> Tuple[Dict[ModuleName, torch.Tensor], Dict[ModuleName, torch.Tensor]]:
    """Collects the maximum values of input and output activations of a specific model.

    Args:
        model: A torch.nn.Module representing the model for which we want to collect the max statistics.
        inputs: An iterator of torch.Tensors representing the inputs to the model.
        target_classes: A tuple of types of torch.nn.Module representing the target classes.
        max_batch_size: An int representing the maximum batch size for input data. Default is 1.

    Returns:
        A tuple of two dictionaries: (max_input_stats, max_output_stats), where:
        max_input_stats: A dictionary of the maximum input activation values for each module of the model.
        max_output_stats: A dictionary of the maximum output activation values for each module of the model.

    This function uses a forward hook to capture the maximum input and output activation values of the specified target_classes.
    The max_batch_size parameter controls the size of the input batches that are passed through the model.

    The function returns two dictionaries containing the maximum input and output activation values for each module of the model,
    respectively. These dictionaries can be used to calculate scaling factors for weight quantization and activation smoothing.
    """
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
        for inputs in tqdm(  # type: ignore[assignment]
            batched(inputs, max_batch_size),
            desc="Collecting stats for SmoothQuant",
        ):
            input = torch.concat(list(inputs))
            model(input.to(device))
    finally:
        for removable in removables:
            removable.remove()
    return max_input_stats, max_output_stats


def get_smoothquant_calibration_dataset(
    data_path: str, data_format: str, data_split: str, seed: int, num_samples: int
):
    """Get SmoothQuant calibration dataset."""
    try:
        if os.path.exists(data_path):
            dataset = load_dataset(data_format, data_files=data_path, split=data_split)
        else:
            dataset = load_dataset(data_path, split="validation")
    except ValueError as err:
        raise CheckpointQuantizationError(f"load_dataset failed. {str(err)}")
    logger.info(
        f"Smoothquant Calibration dataset({data_path}, data_split={data_split}) is successfully loaded!",
    )
    samples = [data["text"] for data in islice(dataset.shuffle(seed), num_samples)]
    return samples


def get_quantized_state_dict(
    quant_results: Iterator[Int8QuantScaleResult],
) -> Dict[str, Any]:
    """Get quantized state dict from quant results."""
    state_dict = {}
    for quant_result in quant_results:
        for filed in fields(quant_result):
            layer_name = filed.name
            layer = cast(Int8QuantScale, getattr(quant_result, layer_name))
            if isinstance(layer, Int8QuantScale):
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
