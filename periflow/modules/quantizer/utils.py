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
from transformers import PreTrainedTokenizer  # type: ignore[import]

from periflow.errors import CheckpointQuantizationError
from periflow.logging import logger
from periflow.modules.quantizer.formatter import (
    Int8QuantScale,
    Int8QuantScaleInput,
    Int8QuantScaleInputTuple,
    Int8QuantScaleResult,
)

ModuleName = str
QUANT_PREFIX = "quant_tf_layer"


def get_torch_data_type(data_type: str):
    """Get torch data type from Enum."""
    if data_type == "fp16":
        return torch.float16
    if data_type == "fp32":
        return torch.float32
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


T = TypeVar("T")


@torch.no_grad()
def collect_max_stats(
    model: torch.nn.Module,
    inputs: Iterator[torch.Tensor],
    target_classes: Tuple[Type[torch.nn.Module], ...],
    max_batch_size: int = 1,
    tqdm_desc: str = "Collecting stats for Smoothing Model",
) -> Tuple[Dict[ModuleName, torch.Tensor], Dict[ModuleName, torch.Tensor]]:
    """Collects the maximum values of input and output activations of a specific model.

    Args:
        model: The model for which we want to collect the max statistics.
        inputs: An iterator of torch.Tensors representing the inputs to the model.
        target_classes: A tuple of types of torch.nn.Module representing the target classes.
        max_batch_size: An int representing the maximum batch size for input data. Default is 1.

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
        for batched_inputs in tqdm(  # type: ignore[assignment]
            batched(inputs, max_batch_size),
            desc=tqdm_desc,
        ):
            batched_input = torch.concat(list(batched_inputs))
            model(batched_input.to(device))
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
) -> Int8QuantScale:
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

    return Int8QuantScale(
        layer_name,
        in_scale=in_scale,
        weight_scale=weight_scale,
        out_scale=out_scale,
        int8_weight=int8_weight,
    )



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
        raise CheckpointQuantizationError(f"load_dataset failed. {str(err)}") from err
    logger.info(
        "Smoothquant Calibration dataset(%s), data_split(%s) is successfully loaded!",
        data_path,
        data_split,
    )
    samples = [data["text"] for data in islice(dataset.shuffle(seed), num_samples)]
    return samples


def get_encoded_samples(
    config: Dict[str, Any], tokenizer: PreTrainedTokenizer
) -> Iterator[Any]:
    """Get encoded samples for SmoothQuant calibration."""
    samples = get_smoothquant_calibration_dataset(
        data_path=config["data_path_or_name"],
        data_format=config["data_format"],
        data_split=config["data_split"],
        seed=config["seed"],
        num_samples=config["num_samples"],
    )
    encoded_samples = (
        tokenizer(
            x,
            return_tensors="pt",
            max_length=config["max_length"],
            truncation=True,
        ).input_ids
        for x in samples
    )
    return encoded_samples


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
