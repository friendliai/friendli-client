# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Savers to save a converted checkpoints into various file types."""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import Dict, Generic, List, TypeVar, Union

import h5py  # type: ignore[import]
import numpy as np
import safetensors.numpy  # type: ignore[import]
import safetensors.torch  # type: ignore[import]
import torch
from typing_extensions import Self

from friendli.enums import CheckpointFileType
from friendli.errors import CheckpointConversionError
from friendli.logging import logger


def get_saver(
    ckpt_file_type: CheckpointFileType,
    output_dir: str,
) -> CheckpointSaver:
    """Create a saver that corresponds to the file type."""
    if ckpt_file_type == CheckpointFileType.HDF5:
        return HDF5Saver(output_dir)
    if ckpt_file_type == CheckpointFileType.SAFETENSORS:
        return SafetensorsSaver(output_dir)
    raise CheckpointConversionError(
        f"Output file type {ckpt_file_type} is not supported."
    )


class CheckpointSaver(AbstractContextManager):
    """Abstract for savers."""

    def __init__(self, output_dir: Union[str, os.PathLike]) -> None:
        """Check that the output file already exists."""
        super().__init__()
        self.output_dir = output_dir

    @abstractmethod
    def save_tensor(self, tensor_id: str, t: Union[np.ndarray, torch.Tensor]) -> None:
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

    def __init__(self, output_dir: str) -> None:
        """Create a HDF5 file."""
        super().__init__(output_dir)
        self._out_f = h5py.File(output_dir + "model.h5", "w")

    def save_tensor(self, tensor_id: str, t: Union[np.ndarray, torch.Tensor]) -> None:
        """Create a group if not exists, and save the tensor in the file."""
        assert isinstance(t, np.ndarray)
        self._out_f[tensor_id] = t

    def close(self) -> None:
        """Close the HDF5 file."""
        self._out_f.close()


T = TypeVar("T")


class SafetensorsSaverInterface(Generic[T]):
    """Interface for saving safetensor format."""

    def get_weight_size(self, tensor: T) -> int:
        """Get total weight size in `Byte` unit."""
        raise NotImplementedError

    def save_file(self, tensor: Dict[str, T], path: str) -> None:
        """Save given tensor to path."""
        raise NotImplementedError


class TorchSafetensorsSaverInterface(SafetensorsSaverInterface[torch.Tensor]):
    """Interface for saving safetensor format."""

    def get_weight_size(self, tensor: torch.Tensor) -> int:
        """Get total weight size in `Byte` unit."""
        return tensor.itemsize * tensor.numel()

    def save_file(self, tensor: Dict[str, torch.Tensor], path: str) -> None:
        """Save given tensor to path."""
        safetensors.torch.save_file(tensor, path)


class NumpySafetensorsSaverInterface(SafetensorsSaverInterface[np.ndarray]):
    """Interface for saving safetensor format."""

    def get_weight_size(self, tensor: np.ndarray) -> int:
        """Get total weight size in `Byte` unit."""
        return tensor.itemsize * tensor.size

    def save_file(self, tensor: Dict[str, np.ndarray], path: str) -> None:
        """Save given tensor to path."""
        safetensors.numpy.save_file(tensor, path)


class UnionSafetensorsSaverInterface(
    SafetensorsSaverInterface[Union[torch.Tensor, np.ndarray]]
):
    """Interface for saving safetensor format."""

    def __init__(self) -> None:
        """Initialize UnionSafetensorsSaverInterface."""
        self._sub_itfcs = {
            np.ndarray: NumpySafetensorsSaverInterface(),
            torch.Tensor: TorchSafetensorsSaverInterface(),
        }
        super().__init__()

    def get_weight_size(self, tensor: Union[torch.Tensor, np.ndarray]) -> int:
        """Get total weight size in `Byte` unit."""
        return self._sub_itfcs[type(tensor)].get_weight_size(tensor)  # type: ignore[attr-defined]

    def save_file(
        self, tensor: Dict[str, Union[torch.Tensor, np.ndarray]], path: str
    ) -> None:
        """Save given tensor to path."""
        for tensor_type, itfc in self._sub_itfcs.items():
            partial_dict = {
                k: v
                for k, v in tensor.items()
                if type(v) == tensor_type  # pylint: disable=unidiomatic-typecheck
            }
            itfc.save_file(partial_dict, path)  # type: ignore[attr-defined]


class SafetensorsSaver(CheckpointSaver):
    """Saver for Safetensors.

    This temporally saves the converted tensors in local memory.
    Then, all of the tensors are saved in the file at a time when close() is called,
    because Safetensors does not support stream saving.
    """

    def __init__(self, output_dir: Union[str, os.PathLike]) -> None:
        """Initialize a saver."""
        super().__init__(output_dir)
        self._output_dir = output_dir
        self._tensors: Dict[str, Union[np.ndarray, torch.Tensor]] = {}
        self._saver: UnionSafetensorsSaverInterface = UnionSafetensorsSaverInterface()

    def save_tensor(self, tensor_id: str, t: Union[np.ndarray, torch.Tensor]) -> None:
        """Save the tensor in the local memory."""
        self._tensors[tensor_id] = t

    def shard_checkpoint(self, max_shard_size: str):
        """Shard the checkpoint with index."""
        # pylint: disable=too-many-locals
        int_max_shard_size = int(max_shard_size[:-2]) * (10**9)
        sharded_tensors: List[Dict[str, Union[np.ndarray, torch.Tensor]]] = [{}]
        last_block_size = 0
        total_size = 0

        for key, weight in self._tensors.items():
            weight_size = self._saver.get_weight_size(weight)
            if (
                last_block_size + weight_size > int_max_shard_size
                and len(sharded_tensors[-1]) > 0
            ):
                sharded_tensors.append({})
                last_block_size = 0

            sharded_tensors[-1][key] = weight
            last_block_size += weight_size
            total_size += weight_size

        if len(sharded_tensors) == 1:
            return {"model.safetensors": sharded_tensors[0]}, None

        weight_map = {}
        shards = {}
        for idx, shard in enumerate(sharded_tensors):
            shard_file = "model.safetensors".replace(
                ".safetensors",
                f"-{idx + 1:05d}-of-{len(sharded_tensors):05d}.safetensors",
            )
            shards[shard_file] = shard
            for key in shard.keys():
                weight_map[key] = shard_file

        metadata = {"total_size": total_size}
        index = {"metadata": metadata, "weight_map": weight_map}
        return shards, index

    def _save_to_file(self) -> None:
        """Save the tensors in the file."""
        logger.info("Saving the converted checkpoint...")

        max_shard_size = "10GB"
        shards, index = self.shard_checkpoint(max_shard_size)

        for shard_file, shard in shards.items():
            self._saver.save_file(shard, os.path.join(self._output_dir, shard_file))

        if index is None:
            path_to_weights = os.path.join(self._output_dir, "model.safetensors")
            logger.info("Model weights saved in (%s)", path_to_weights)
        else:
            save_index_file = os.path.join(
                self._output_dir, "model.safetensors.index.json"
            )
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                "The model is bigger than the maximum size per checkpoint %s "
                " and is going to be split in %s checkpoint shards. You can find "
                "where each parameters has been saved in the index located at (%s).",
                max_shard_size,
                str(len(shards)),
                save_index_file,
            )

    def close(self) -> None:
        """Save the tensors in the file."""
        self._save_to_file()
