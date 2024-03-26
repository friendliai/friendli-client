# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli CLI File System Management Utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union

friendli_directory = Path.home() / ".friendli"


def get_friendli_directory() -> Path:
    """Get a hidden path to Friendli config directory."""
    friendli_directory.mkdir(exist_ok=True)
    return friendli_directory


def storage_path_to_local_path(storage_path: str, source_path: Path) -> Path:
    """Translate a cloud storage path to the local file system path."""
    return Path(strip_storage_path_prefix(str(source_path / storage_path)))


def get_file_size(file_path: Union[str, Path]) -> int:
    """Calculate a file size in bytes.

    Args:
        file_path (Union[str, Path]): Path to the target file.

    Returns:
        int: The size of a file.

    """
    if isinstance(file_path, str):
        return os.stat(file_path).st_size
    return file_path.stat().st_size


def strip_storage_path_prefix(path: str) -> str:
    """Strip a prefix about auxiliary checkpoint info from the path.

    Args:
        path (str): Actual checkpoint storage path. The path may starts with the prefix
            that contain the iteration number and distributed training configuration.

    Returns:
        str: Path without the prefix.

    """
    return re.sub(
        pattern=r"iter_\d{7}/mp\d{3}-\d{3}pp\d{3}-\d{3}/",
        repl="",
        string=path,
    )


def attach_storage_path_prefix(
    path: Union[str, Path],
    iteration: int,
    mp_rank: int,
    mp_degree: int,
    pp_rank: int,
    pp_degree: int,
) -> str:
    """Attach a filename prefix to mark up iteration and dist info.

    Args:
        path (Union[str, Path]): The relative path to file.
        iteration (int): Checkpoint iteration number.
        mp_rank (int): Model parallelism(a.k.a. tensor parallelism) rank.
        mp_degree (int): Model parallelism(a.k.a. tensor parallelism) degree.
        pp_rank (int): Pipelined model parallelism rank.
        pp_degree (int): Pipelined model parallelism degree.

    Returns:
        str: Storage path with the prefix

    """
    # pylint: disable=line-too-long
    return f"iter_{iteration:07d}/mp{mp_rank:03d}-{mp_degree:03d}pp{pp_rank:03d}-{pp_degree:03d}/{path}"
