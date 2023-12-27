# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Savers to save a converted checkpoints into various file types."""

from __future__ import annotations

import os
from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import Dict, Union

import h5py  # type: ignore[import]
import numpy as np
import safetensors.numpy  # type: ignore[import]
from typing_extensions import Self

from friendli.enums import CheckpointFileType
from friendli.errors import CheckpointConversionError
from friendli.logging import logger


def get_saver(
    ckpt_file_type: CheckpointFileType, output_path: Union[str, os.PathLike]
) -> CheckpointSaver:
    """Create a saver that corresponds to the file type."""
    if ckpt_file_type == CheckpointFileType.HDF5:
        return HDF5Saver(output_path)
    if ckpt_file_type == CheckpointFileType.SAFETENSORS:
        return SafetensorsSaver(output_path)
    raise CheckpointConversionError(
        f"Output file type {ckpt_file_type} is not supported."
    )


class CheckpointSaver(AbstractContextManager):
    """Abstract for savers."""

    def __init__(self, output_path: Union[str, os.PathLike]) -> None:
        """Check that the output file already exists."""
        super().__init__()
        self._output_path = output_path

    @abstractmethod
    def save_tensor(self, tensor_id: str, t: np.ndarray) -> None:
        """Save the tensor in the file."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the output checkpoint file."""
        raise NotImplementedError

    def __enter__(self) -> Self:
        """Enter for context manager."""
        return self

    def __exit__(self, *exc) -> None:
        """Exit for context manager."""
        self.close()


class HDF5Saver(CheckpointSaver):
    """Saver for HDF5."""

    def __init__(self, output_path: Union[str, os.PathLike]) -> None:
        """Create a HDF5 file."""
        super().__init__(output_path)
        self._out_f = h5py.File(output_path, "w")

    def save_tensor(self, tensor_id: str, t: np.ndarray) -> None:
        """Create a group if not exists, and save the tensor in the file."""
        self._out_f[tensor_id] = t

    def close(self) -> None:
        """Close the HDF5 file."""
        self._out_f.close()


class SafetensorsSaver(CheckpointSaver):
    """Saver for Safetensors.

    This temporally saves the converted tensors in local memory.
    Then, all of the tensors are saved in the file at a time when close() is called,
    because Safetensors does not support stream saving.
    """

    def __init__(self, output_path: Union[str, os.PathLike]) -> None:
        """Initialize a saver."""
        super().__init__(output_path)
        self._output_path = output_path
        self._tensors: Dict[str, np.ndarray] = {}

    def save_tensor(self, tensor_id: str, t: np.ndarray) -> None:
        """Save the tensor in the local memory."""
        self._tensors[tensor_id] = t

    def _save_to_file(self) -> None:
        """Save the tensors in the file."""
        logger.info("Saving the converted checkpoint...")
        safetensors.numpy.save_file(self._tensors, self._output_path)

    def close(self) -> None:
        """Save the tensors in the file."""
        self._save_to_file()
