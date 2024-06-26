# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Quantizer Utils."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
)

import datasets  # type: ignore[import]
import torch
from accelerate import cpu_offload_with_hook  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (  # type: ignore
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from friendli.errors import (
    InvalidConfigError,
    NotFoundError,
    QuantizationError,
    TokenizerNotFoundError,
)
from friendli.logging import logger
from friendli.modules.quantizer_v2.enums import ModelDataType
from friendli.modules.quantizer_v2.schema.config import CalibrationDatasetConfig
from friendli.modules.quantizer_v2.schema.data import (
    ModuleName,
    WeightActQuantResult,
    WeightOnlyQuantResult,
)


def get_tokenizer(
    model_name_or_path: str,
    *,
    cache_dir: Optional[str] = None,
) -> PreTrainedTokenizer:
    """Try to get tokenizer of a pretrained model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except OSError as exc:
        raise TokenizerNotFoundError(str(exc)) from exc

    if not tokenizer.is_fast:
        raise TokenizerNotFoundError(
            "This model does not support Friendli-compatible tokenizer"
        )

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def save_tokenizer(
    model_name_or_path: str,
    *,
    cache_dir: Optional[str] = None,
    save_dir: str,
) -> Tuple[str, ...]:
    """Try to save `tokenizer.json` of a pretrained model."""
    if not os.path.isdir(save_dir):
        raise NotFoundError(f"Directory '{save_dir}' is not found.")

    tokenizer = get_tokenizer(model_name_or_path, cache_dir=cache_dir)
    saved_file_paths = tokenizer.save_pretrained(save_directory=save_dir)
    tokenizer_json_path = None
    for path in saved_file_paths:
        if "tokenizer.json" == os.path.basename(path):
            tokenizer_json_path = path
            break

    if tokenizer_json_path is None:
        raise TokenizerNotFoundError(
            "This model has the Friendli-compatible tokenizer implementation, but "
            "'tokenizer.json' file is not found."
        )
    return saved_file_paths


def get_model_pretrained_config(
    model_name_or_path: str, model_output_path: str, cache_dir: Optional[str] = None
) -> PretrainedConfig:
    """Get HuggingFace model configs."""
    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, trust_remote_code=True
        )
    except OSError as exc:  # from AutoConfig.from_pretrained()
        config_dir = Path(model_name_or_path)
        model_output_dir = Path(model_output_path).parent
        if config_dir.exists() and model_output_dir.absolute() == config_dir.absolute():
            raise NotFoundError(
                f"'output_dir' ({model_output_dir.as_posix()}) and "
                f"'model_name_or_path' ({model_name_or_path}) are the same. "
                "In such a case, checkpoints should be prepared in 'output_dir'."
            ) from exc
        raise NotFoundError(str(exc)) from exc

    return config


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
    model: PreTrainedModel,
    device: str,
    calib_dataloader: DataLoader,
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
        for inputs in tqdm(calib_dataloader, desc=tqdm_desc):
            model(inputs.to(device))
    finally:
        for removable in removables:
            removable.remove()
    return max_input_stats(), max_output_stats()


def convert_tensor_to_quant_dtype(
    param: torch.Tensor,
    quant_dtype: ModelDataType,
) -> torch.Tensor:
    """Convert tensor format to the given data type.

    Args:
        param (torch.Tensor): The tensor to be converted.
        data_type (ModelDataType): The data type of the tensor.

    Returns:
        torch.Tensor: The converted tensor.

    """
    assert quant_dtype in [ModelDataType.INT4, ModelDataType.INT8]
    if quant_dtype is ModelDataType.INT4:
        pack_num = 8 // 4
        int4_param = torch.zeros(
            (param.shape[0], param.shape[1] // pack_num),
            dtype=torch.uint8,
            device=param.device,
        )
        for col in range(int4_param.shape[1]):
            for i in range(pack_num):
                int4_param[:, col] |= param[:, col * pack_num + i] << (i * 4)
        param = int4_param.to(torch.int8)

    elif quant_dtype is ModelDataType.INT8:
        param = param.to(torch.int8)

    return param.detach().to("cpu")


@torch.no_grad()
def get_weight_act_quant_scales(
    model: PreTrainedModel,
    layer_names: List[str],
    max_input_stats: Dict[ModuleName, torch.Tensor],
    device: str = "cpu",
    quant_dtype: ModelDataType = ModelDataType.INT8,
    quant_scale_dtype: ModelDataType = ModelDataType.FP32,
) -> List[WeightActQuantResult]:
    """Get the quantization scales and int8 weight for a specific layer."""
    input_max = torch.concat([max_input_stats[name] for name in layer_names])
    target_weights = [model.get_submodule(name).weight for name in layer_names]
    target_weight = torch.concat(target_weights)

    max_val = 2 ** (8 - 1) - 1
    min_val = -(2 ** (8 - 1))

    act_scale = float(input_max.detach().abs().max().item()) / float(max_val)
    weight_scale = float(target_weight.detach().abs().max().item()) / float(max_val)

    q_weights = [
        (
            convert_tensor_to_quant_dtype(
                (weight.detach().float() / weight_scale).clip(min_val, max_val),
                quant_dtype,
            ).to(device)
        )
        for weight in target_weights
    ]
    quant_scale_torch_dtype = get_torch_data_type(quant_scale_dtype)
    return [
        WeightActQuantResult(
            act_scale=torch.tensor(act_scale, dtype=quant_scale_torch_dtype),
            weight_scale=torch.tensor(weight_scale, dtype=quant_scale_torch_dtype),
            q_weight=q_weight,
            q_group_size=-1,
            zero_point=None,
        )
        for _, q_weight in zip(layer_names, q_weights)
    ]


def get_weight_only_quant_scales(
    model: PreTrainedModel,
    layer_names: List[str],
    quant_dtype: ModelDataType,
    quant_scale_dtype: ModelDataType,
    q_group_size: int = -1,
    use_symmetric: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> List[WeightOnlyQuantResult]:
    """Return the quantization scales of weight for a specific layer."""
    # pylint: disable=too-many-locals
    assert quant_dtype in [ModelDataType.INT4, ModelDataType.INT8]
    q_bit = 4 if quant_dtype == ModelDataType.INT4 else 8
    target_weights = [model.get_submodule(name).weight for name in layer_names]
    org_w_shape = target_weights[0].shape  # [OutDim, InDim]
    w = torch.concat(target_weights)

    if q_group_size != -1:
        w = w.reshape(-1, q_group_size)  # [OutDim x num_groups, group_size]

    if use_symmetric:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_int = 2 ** (q_bit - 1) - 1
        min_int = -(2 ** (q_bit - 1))
        scales = (max_val / float(max_int)).clamp(min=1e-5)
        zeros = torch.zeros_like(max_val)
    else:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**q_bit - 1
        min_int = 0

        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    q_weights = [
        convert_tensor_to_quant_dtype(
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
            .reshape(org_w_shape)
            .detach(),
            quant_dtype,
        ).to(device)
        for w in target_weights
    ]
    quant_scale_torch_dtype = get_torch_data_type(quant_scale_dtype)
    scales = (
        scales.view(org_w_shape[0], -1).detach().transpose(0, 1).to(device)
    )  # [num_groups, OutDim]
    zeros = (
        zeros.view(org_w_shape[0], -1).detach().transpose(0, 1).to(device)
    )  # [num_groups, OutDim]

    if q_group_size == -1:
        scales = scales.squeeze(0)
        zeros = zeros.squeeze(0)

    return [
        WeightOnlyQuantResult(
            zero_point=None if use_symmetric else zeros.to(quant_scale_torch_dtype),
            q_group_size=q_group_size,
            weight_scale=scales.to(quant_scale_torch_dtype),
            q_weight=q_weight,
        )
        for q_weight in q_weights
    ]


def get_model_dtype(torch_dtype: torch.dtype) -> ModelDataType:
    """Get torch data type from Enum."""
    if torch_dtype == torch.float16:
        return ModelDataType.FP16
    if torch_dtype == torch.float32:
        return ModelDataType.FP32
    if torch_dtype == torch.bfloat16:
        return ModelDataType.BF16
    raise QuantizationError(f"{torch_dtype} is not valid dtype for hf model dtype.")


def get_torch_data_type(data_type: str) -> torch.dtype:
    """Get torch data type from Enum."""
    if data_type == ModelDataType.FP16:
        return torch.float16
    if data_type == ModelDataType.FP32:
        return torch.float32
    if data_type == ModelDataType.BF16:
        return torch.bfloat16
    raise QuantizationError(
        f"Can't not converted original param to {data_type}. Only FP16, FP32, BF16 are supported."
    )


def send_model_to_device(
    model: PreTrainedModel,
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
