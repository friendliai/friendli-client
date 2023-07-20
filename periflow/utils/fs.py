# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CLI File System Management Utilities."""

from __future__ import annotations

import os
import re
import zipfile
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from functools import wraps
from io import BufferedReader
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pathspec  # type: ignore
import requests
import typer
from dateutil.tz import tzlocal
from requests import Request, Session
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from periflow.utils.format import secho_error_and_exit
from periflow.utils.request import DEFAULT_REQ_TIMEOUT

# The actual hard limit of a part size is 5 GiB, and we use 200 MiB part size.
# See https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html.
S3_MPU_PART_MAX_SIZE = 200 * 1024 * 1024  # 200 MiB
S3_UPLOAD_SIZE_LIMIT = 5 * 1024 * 1024 * 1024  # 5 GiB

periflow_directory = Path.home() / ".periflow"


def get_periflow_directory() -> Path:
    """Get a hidden path to PeriFlow config directory."""
    periflow_directory.mkdir(exist_ok=True)
    return periflow_directory


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


def storage_path_to_local_path(storage_path: str, source_path: Path) -> str:
    """Translate a cloud storage path to the local file system path."""
    return strip_storage_path_prefix(str(source_path / storage_path))


def get_file_info(storage_path: str, source_path: Path) -> Dict[str, Any]:
    """Get file metadata."""
    loacl_path = storage_path_to_local_path(storage_path, source_path)
    return {
        "name": os.path.basename(storage_path),
        "path": storage_path,
        "mtime": datetime.fromtimestamp(
            os.stat(loacl_path).st_mtime, tz=tzlocal()
        ).isoformat(),
        "size": os.stat(loacl_path).st_size,
    }


def get_content_size(url: str) -> int:
    """Get download content size."""
    response = requests.get(url, stream=True, timeout=DEFAULT_REQ_TIMEOUT)
    if response.status_code != 200:
        secho_error_and_exit("Failed to download (invalid url)")
    return int(response.headers["Content-Length"])


def download_range(url: str, start: int, end: int, output: str, ctx: tqdm) -> None:
    """Download a specific part of a file from the URL."""
    headers = {"Range": f"bytes={start}-{end}"}
    response = requests.get(
        url, headers=headers, stream=True, timeout=DEFAULT_REQ_TIMEOUT
    )

    with open(output, "wb") as f:
        wrapped_object = CallbackIOWrapper(ctx.update, f, "write")
        for part in response.iter_content(1024):
            wrapped_object.write(part)


def download_file_simple(url: str, out: str, content_length: int) -> None:
    """Download a file without parallelism."""
    response = requests.get(url, stream=True, timeout=DEFAULT_REQ_TIMEOUT)
    with tqdm.wrapattr(
        open(out, "wb"), "write", miniters=1, total=content_length
    ) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)


def download_file_parallel(
    url: str, out: str, content_length: int, chunk_size: int = 1024 * 1024 * 4
) -> None:
    """Download a file in parallel."""
    chunks = range(0, content_length, chunk_size)

    temp_out_prefix = os.path.join(os.path.dirname(out), f".{os.path.basename(out)}")

    try:
        with tqdm(
            total=content_length, unit="B", unit_scale=True, unit_divisor=1024
        ) as t:
            with ThreadPoolExecutor() as executor:
                futs = [
                    executor.submit(
                        download_range,
                        url,
                        start,
                        start + chunk_size - 1,
                        f"{temp_out_prefix}.part{i}",
                        t,
                    )
                    for i, start in enumerate(chunks)
                ]
                wait(futs, return_when=FIRST_EXCEPTION)

        # Merge partitioned files
        with open(out, "wb") as f:
            for i in range(len(chunks)):
                chunk_path = f"{temp_out_prefix}.part{i}"
                with open(chunk_path, "rb") as chunk_f:
                    f.write(chunk_f.read())

                os.remove(chunk_path)
    finally:
        # Clean up zombie temporary partitioned files
        for i in range(len(chunks)):
            chunk_path = f"{temp_out_prefix}.part{i}"
            if os.path.isfile(chunk_path):
                os.remove(chunk_path)


def download_file(url: str, out: str) -> None:
    """Download a file from the URL."""
    file_size = get_content_size(url)

    # Create directory if not exists
    dirpath = os.path.dirname(out)
    try:
        os.makedirs(dirpath, exist_ok=True)
    except OSError as exc:
        secho_error_and_exit(
            f"Cannot create directory({dirpath}) to download file: {exc!r}"
        )

    if file_size >= 16 * 1024 * 1024:
        download_file_parallel(url, out, file_size)
    else:
        download_file_simple(url, out, file_size)


class FileSizeType(Enum):
    """File size type."""

    LARGE = "LARGE"
    SMALL = "SMALL"


def _filter_large_files(paths: List[str]) -> List[str]:
    return [path for path in paths if get_file_size(path) >= S3_UPLOAD_SIZE_LIMIT]


def _filter_small_files(paths: List[str]) -> List[str]:
    res = []
    for path in paths:
        size = get_file_size(path)
        if size == 0:
            # NOTE: S3 does not support file uploading for 0B size files.
            typer.secho(
                f"Skip uploading file ({path}) with size 0B.", fg=typer.colors.RED
            )
            continue
        if size < S3_UPLOAD_SIZE_LIMIT:
            res.append(path)
    return res


def filter_files_by_size(paths: List[str], size_type: FileSizeType) -> List[str]:
    """Filter files by size.

    Args:
        paths (List[str]): List of paths.
        size_type (FileSizeType): File size type to filter.

    Returns:
        List[str]: Filtered file paths.

    """
    handler_map = {
        FileSizeType.LARGE: _filter_large_files,
        FileSizeType.SMALL: _filter_small_files,
    }
    return handler_map[size_type](paths)


def expand_paths(path: Path, size_type: FileSizeType) -> List[str]:
    """Expand all file paths from the source path.

    Args:
        path (Path): The source path to expand files
        size_type (FileSizeType): The file size type to filter

    Returns:
        List[str]: A list of file paths from the source path
    """
    if path.is_file():
        paths = [str(path)]
        paths = filter_files_by_size(paths, size_type)
    else:
        paths = [str(p) for p in path.rglob("*")]
        paths = filter_files_by_size(paths, size_type)

    # Filter files only
    paths = [path for path in paths if os.path.isfile(path)]

    return paths


def upload_file(file_path: str, url: str, ctx: tqdm) -> None:
    """Upload a file to the URL."""
    try:
        with open(file_path, "rb") as f:
            fileno = f.fileno()
            total_file_size = os.fstat(fileno).st_size
            if total_file_size == 0:
                typer.secho(
                    f"The file with 0B size ({file_path}) is skipped.",
                    fg=typer.colors.RED,
                )
                return

            wrapped_object = CallbackIOWrapper(ctx.update, f, "read")
            with Session() as s:
                req = Request("PUT", url, data=wrapped_object)
                prep = req.prepare()
                prep.headers["Content-Length"] = str(
                    total_file_size
                )  # necessary to use ``CallbackIOWrapper``
                response = s.send(prep)
            if response.status_code != 200:
                secho_error_and_exit(
                    f"Failed to upload file ({file_path}): {response.content!r}"
                )
    except FileNotFoundError:
        secho_error_and_exit(f"{file_path} is not found.")


def get_file_size(file_path: str, prefix: Optional[str] = None) -> int:
    """Calculate a file size in bytes.

    Args:
        file_path (str): Path to the target file.
        prefix (Optional[str], optional): Attach the prefix to the path.

    Returns:
        int: The size of a file.
    """
    if prefix is not None:
        file_path = os.path.join(prefix, file_path)

    return os.stat(file_path).st_size


def get_total_file_size(file_paths: List[str], prefix: Optional[str] = None) -> int:
    """Get total file size in the paths."""
    return sum(get_file_size(file_path, prefix) for file_path in file_paths)


class CustomCallbackIOWrapper(CallbackIOWrapper):
    """Cutom callback IO wrapper."""

    def __init__(self, callback, stream, method="read", chunk_size=None) -> None:
        """Initialize custom callback IO wrapper."""
        # Wrap a file-like object's `read` or `write` to report data length to the `callback`.
        super().__init__(callback, stream, method)
        self._chunk_size = chunk_size
        self._cursor = 0

        func = getattr(stream, method)
        if method == "write":

            @wraps(func)
            def write(data, *args, **kwargs):
                res = func(data, *args, **kwargs)
                callback(len(data))
                return res

            self.wrapper_setattr("write", write)
        elif method == "read":

            @wraps(func)
            def read(*args, **kwargs):
                assert chunk_size is not None
                if self._cursor >= chunk_size:
                    self._cursor = 0
                    return None

                data = func(*args, **kwargs)
                data_size = len(data)  # default to 8 KiB
                callback(data_size)
                self._cursor += data_size
                return data

            self.wrapper_setattr("read", read)
        else:
            raise KeyError("Can only wrap read/write methods")


# pylint: disable=too-many-locals
def upload_part(
    file_path: str,
    chunk_index: int,
    part_number: int,
    upload_url: str,
    ctx: tqdm,
    is_last_part: bool,
) -> Dict[str, Any]:
    """Upload a specific part of the multipart upload payload.

    Args:
        file_path (str): Path to file to upload
        chunk_index (int): Chunk index to upload
        part_number (int): Part number of the multipart upload
        upload_url (str): A presigned URL for the multipart upload
        ctx (tqdm): tqdm context to update the progress
        is_last_part (bool): Whether this part is the last part of the payload.

    Returns:
        Dict[str, Any]: _description_
    """
    with open(file_path, "rb") as f:
        fileno = f.fileno()
        total_file_size = os.fstat(fileno).st_size
        cursor = chunk_index * S3_MPU_PART_MAX_SIZE
        f.seek(cursor)
        chunk_size = min(S3_MPU_PART_MAX_SIZE, total_file_size - cursor)
        wrapped_object = CustomCallbackIOWrapper(ctx.update, f, "read", chunk_size)
        with Session() as s:
            req = Request("PUT", upload_url, data=wrapped_object)
            prep = req.prepare()
            prep.headers["Content-Length"] = str(chunk_size)
            response = s.send(prep)
        response.raise_for_status()

        if is_last_part:
            assert not f.read(
                S3_MPU_PART_MAX_SIZE
            ), "Some parts of your data is not uploaded. Please try again."

    etag = response.headers["ETag"]
    return {
        "etag": etag,
        "part_number": part_number,
    }


def strip_storage_path_prefix(path: str) -> str:
    """Strip a prefix about auxiliary checkpoint info from the path.

    Args:
        path (str): Actual checkpoint storage path. The path may starts with the prefix
                    that contain the iteration number and distributed training configuration

    Returns:
        str: Path without the prefix
    """
    return re.sub(
        pattern=r"iter_\d{7}/mp\d{3}-\d{3}pp\d{3}-\d{3}/",
        repl="",
        string=path,
    )


def attach_storage_path_prefix(
    path: str,
    iteration: int,
    mp_rank: int,
    mp_degree: int,
    pp_rank: int,
    pp_degree: int,
) -> str:
    """Attach a filename prefix to mark up iteration and dist info.

    Args:
        path (str): The relative path to file
        iteration (int): Checkpoint iteration number
        mp_rank (int): Model parallelism(a.k.a. tensor parallelism) rank
        mp_degree (int): Model parallelism(a.k.a. tensor parallelism) degree
        pp_rank (int): Pipelined model parallelism rank
        pp_degree (int): Pipelined model parallelism degree

    Returns:
        str: Storage path with the prefix

    """
    # pylint: disable=line-too-long
    return f"iter_{iteration:07d}/mp{mp_rank:03d}-{mp_degree:03d}pp{pp_rank:03d}-{pp_degree:03d}/{path}"
