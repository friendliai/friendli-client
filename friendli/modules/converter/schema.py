# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Checkpoint Converter Schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch

from friendli.enums import CheckpointDataType


@dataclass
class ConvertInfo:
    """Dataclass for convert information of the parameter in huggingface checkpoint.

    Args:
        param_names(List[str]): List of parameter names in the huggingface checkpoint.
        data_type(CheckpointDataType): Data type of the parameter.
        converted_name(str): Name of the converted parameter.
        reshape_fn(Callable[[List[torch.tensor]], np.ndarray]):
            Function to reshape the tensor from the huggignface checkpoint.
    """

    param_names: List[str]
    data_type: CheckpointDataType
    converted_name: str
    reshape_fn: Callable[[List[torch.Tensor]], torch.Tensor]
