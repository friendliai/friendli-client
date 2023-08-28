# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint Converter Utils."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import transformers  # type: ignore[import]

from periflow.enums import CheckpointDataType
from periflow.errors import (
    CheckpointConversionError,
    NotFoundError,
    TokenizerNotFoundError,
)
from periflow.logging import logger


def convert_to_gpt_j_params(param: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Convert weight or bias tensor with rotary embedding to gpt-j format.

    Args:
        param (torch.Tensor): Target tensor to convert. Shape must be (num_heads, head_size, ...)
        rotary_dim (int): Degree of rotary embedding

    Returns:
        Torch tensor that heads are rotated.

    Raises:
        CheckpointConversionError: If arguments do not satisfy the requirements.

    """
    if param.ndim < 2:
        raise CheckpointConversionError(
            "Tensor dimension should be greater or equal than 2 for rotary conversion, "
            f"but got {param.ndim}"
        )

    head_size = param.shape[1]
    if rotary_dim > head_size:
        raise CheckpointConversionError(
            f"'rotary_dim' ({rotary_dim}) should be less or equal than 'head_size' ({head_size})"
        )

    param_rot = param[:, :rotary_dim]
    param_pass = param[:, rotary_dim:]

    origin_shape = param_rot.shape
    param_rot_1 = param_rot[:, : rotary_dim // 2]
    param_rot_2 = param_rot[:, rotary_dim // 2 :]
    param_rot = torch.stack((param_rot_1, param_rot_2), dim=2).reshape(*origin_shape)

    return torch.cat((param_rot, param_pass), dim=1)


def get_tensor_from_state_dict(
    state_dict: Dict[str, Any], tensor_name: str
) -> torch.Tensor:
    """Get the tensor whose name is 'tensor_name' from 'state_dict'.

    Args:
        state_dict (Dict[str, Any]): Model checkpoint's state_dict.
        tensor_name (str): Name of tensor to get.

    Returns:
        Corresponding torch Tensor.

    Raises:
        CheckpointConversionError: If 'tensor_name' does not exist in 'state_dict'

    """
    if tensor_name not in state_dict:
        raise CheckpointConversionError(
            f"Cannot find '{tensor_name}' in the model checkpoint"
        )

    return state_dict[tensor_name]


def convert_tensor_to_np_array(
    param: torch.Tensor,
    data_type: CheckpointDataType,
) -> np.ndarray:
    """Convert tensor to numpy ndarray.

    Args:
        param (torch.Tensor): The tensor to be converted.
        data_type (CheckpointDataType): The data type of the tensor.

    Returns:
        np.ndarray: The converted numpy ndarray from the tensor.

    """
    dtype_map = {
        CheckpointDataType.BF16: torch.bfloat16,
        CheckpointDataType.FP16: torch.float16,
        CheckpointDataType.FP32: torch.float32,
    }

    dtype = dtype_map[data_type]

    if data_type is CheckpointDataType.BF16:
        return (
            param.detach().to(dtype).view(dtype=torch.float16).numpy().view(np.uint16)
        )

    return param.detach().to(dtype).numpy()


def get_tokenizer(
    model_name_or_path: str,
    *,
    cache_dir: Optional[str] = None,
) -> transformers.AutoTokenizer:
    """Try to get tokenizer of a pretrained model."""
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except OSError as exc:
        raise TokenizerNotFoundError(str(exc)) from exc

    if not tokenizer.is_fast:
        raise TokenizerNotFoundError(
            "This model does not support PeriFlow-compatible tokenizer"
        )

    return tokenizer


def save_tokenizer(
    model_name_or_path: str,
    *,
    cache_dir: Optional[str] = None,
    save_dir: str,
) -> None:
    """Try to save `tokenizer.json` of a pretrained model."""
    if not os.path.isdir(save_dir):
        raise NotFoundError(f"Directory '{save_dir}' is not found.")

    tokenizer = get_tokenizer(model_name_or_path, cache_dir=cache_dir)
    saved_file_paths = tokenizer.save_pretrained(save_directory=save_dir)

    tokenizer_json_path = None
    for path in saved_file_paths:
        if "tokenizer.json" == os.path.basename(path):
            tokenizer_json_path = path
        else:
            # Remove unnecessary files.
            try:
                os.remove(path)
            except FileNotFoundError:
                logger.warn(
                    "Tried to delete unnecessary tokenizer file %s but the file "
                    "is not found.",
                    path,
                )

    if tokenizer_json_path is None:
        raise TokenizerNotFoundError(
            "This model has the PeriFlow-compatible tokenizer implementation, but "
            "'tokenizer.json' file is not found."
        )
