# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli CLI File System Management Utilities."""

from __future__ import annotations

import os
import re
import zipfile
from contextlib import contextmanager
from datetime import datetime
from io import BufferedReader
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

import pathspec  # type: ignore
import typer
from dateutil.tz import tzlocal

friendli_directory = Path.home() / ".friendli"


def get_friendli_directory() -> Path:
    """Get a hidden path to Friendli config directory."""
    friendli_directory.mkdir(exist_ok=True)
    return friendli_directory


@contextmanager
def zip_dir(
    base_path: Path, target_files: List[Path], zip_path: Path
) -> Iterator[BufferedReader]:
    """Zip direcotry.

    Args:
        base_path (Path): Base directory path. The target files will be located at the
            path related to the base path when the zip file is unzipped.
        target_files (List[Path]): All target files to zip.
        zip_path (Path): Path to save zip file.

    Yields:
        BufferedReader: Opened zip file object.

    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zip_file:
        for file in target_files:
            zip_file.write(file, file.relative_to(base_path.parent))
    typer.secho("Uploading workspace directory...", fg=typer.colors.MAGENTA)
    try:
        yield zip_path.open("rb")
    finally:
        zip_path.unlink()


def get_workspace_files(dir_path: Path) -> List[Path]:
    """Get workspace file paths.

    Args:
        dir_path (Path): Path to the workspace directory.

    Returns:
        List[Path]: A list of paths to the file inside the workspace.

    """
    ignore_file = dir_path / ".pfignore"
    all_files = set(x for x in dir_path.rglob("*") if x.is_file() and x != ignore_file)

    if not ignore_file.exists():
        return list(all_files)

    with open(ignore_file, "r", encoding="utf-8") as f:
        ignore_patterns = f.read()
    spec = pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern, ignore_patterns.splitlines()
    )
    matched_files = set(dir_path / x for x in spec.match_tree_files(dir_path))  # type: ignore
    return list(all_files.difference(matched_files))


def storage_path_to_local_path(storage_path: str, source_path: Path) -> Path:
    """Translate a cloud storage path to the local file system path."""
    return Path(strip_storage_path_prefix(str(source_path / storage_path)))


def get_file_info(storage_path: str, source_path: Path) -> Dict[str, Any]:
    """Get file metadata."""
    local_path = storage_path_to_local_path(storage_path, source_path)
    stat = local_path.stat()
    return {
        "name": os.path.basename(storage_path),
        "path": storage_path,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=tzlocal()).isoformat(),
        "size": stat.st_size,
    }


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
