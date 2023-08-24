# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Object Transfer Utils."""

# pylint: disable=too-many-locals, too-many-arguments


from __future__ import annotations

import os
import socket
from concurrent.futures import FIRST_EXCEPTION, Executor, ThreadPoolExecutor, wait
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from requests import Request, Session
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from urllib3.exceptions import ReadTimeoutError

from periflow.errors import (
    InvalidPathError,
    MaxRetriesExceededError,
    NotFoundError,
    TransferError,
)
from periflow.logging import logger
from periflow.schema.resource.v1.transfer import (
    MultipartUploadTask,
    UploadedPartETag,
    UploadTask,
)
from periflow.utils.fs import get_file_size, storage_path_to_local_path
from periflow.utils.request import DEFAULT_REQ_TIMEOUT

KiB = 1024
MiB = KiB * KiB
GiB = MiB * KiB
IO_CHUNK_SIZE = 256 * KiB
S3_MULTIPART_THRESHOLD = 16 * MiB
S3_MAX_PART_SIZE = 16 * MiB
S3_RETRYABLE_DOWNLOAD_ERRORS = (
    socket.timeout,
    ConnectionError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    ReadTimeoutError,
)
MAX_RETRIES = 5


class DownloadManager:
    """Download manager."""

    def __init__(
        self,
        io_chunk_size: int = IO_CHUNK_SIZE,
        multipart_threshold: int = S3_MULTIPART_THRESHOLD,
        max_part_size: int = S3_MAX_PART_SIZE,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        """Initializes DownloadManager."""
        self._io_chunk_size = io_chunk_size
        self._multipart_threshold = multipart_threshold
        self._max_part_size = max_part_size
        self._max_retries = max_retries

    def download_file(self, url: str, out: str) -> None:
        """Download a file from the URL."""
        file_size = self._get_content_size(url)

        # Create directory if not exists
        dirpath = os.path.dirname(out)
        try:
            os.makedirs(dirpath, exist_ok=True)
        except OSError as exc:
            raise InvalidPathError(
                f"Cannot create directory({dirpath}) to download file: {exc!r}"
            ) from exc

        if file_size < self._multipart_threshold:
            self._download_file_sequential(url, out, file_size)
        else:
            self._download_file_parallel(url, out, file_size)

    def _download_file_sequential(
        self, url: str, out: str, content_length: int
    ) -> None:
        """Downloads a file sequentially."""
        response = requests.get(url, stream=True, timeout=DEFAULT_REQ_TIMEOUT)

        with tqdm.wrapattr(
            open(out, "wb"),
            "write",
            desc=os.path.basename(out),
            miniters=1,
            total=content_length,
        ) as fout:
            for chunk in response.iter_content(IO_CHUNK_SIZE):
                fout.write(chunk)

    def _download_file_parallel(self, url: str, out: str, content_length: int) -> None:
        """Downloads a file in parallel."""
        chunks = range(0, content_length, self._max_part_size)

        temp_out_prefix = os.path.join(
            os.path.dirname(out), f".{os.path.basename(out)}"
        )

        try:
            with tqdm(
                desc=os.path.basename(out),
                total=content_length,
                unit="B",
                unit_scale=True,
                unit_divisor=KiB,
            ) as pbar:
                with ThreadPoolExecutor() as executor:
                    futs = {
                        executor.submit(
                            self._download_range,
                            url,
                            start,
                            start + self._max_part_size - 1,
                            f"{temp_out_prefix}.part{i}",
                            pbar,
                        )
                        for i, start in enumerate(chunks)
                    }
                    not_done = futs
                    try:
                        while not_done:
                            done, not_done = wait(
                                not_done, timeout=1, return_when=FIRST_EXCEPTION
                            )
                            for fut in done:
                                fut.result()
                    except KeyboardInterrupt as exc:
                        logger.warn(
                            "Keyboard interrupted. Wait a few seconds for the shutdown."
                        )
                        # TODO: py38 does not support cancel_futures option. Add cancel_futures=True after deprecation.
                        for fut in not_done:
                            fut.cancel()
                        executor.shutdown(wait=False)
                        raise exc

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

    def _download_range(
        self, url: str, start: int, end: int, output: str, ctx: tqdm
    ) -> None:
        """Download a specific part of a file from the URL."""
        headers = {"Range": f"bytes={start}-{end}"}
        final_exc = None
        for i in range(self._max_retries):
            try:
                response = requests.get(
                    url, headers=headers, stream=True, timeout=DEFAULT_REQ_TIMEOUT
                )
                final_exc = None
                break
            except S3_RETRYABLE_DOWNLOAD_ERRORS as exc:
                logger.debug(
                    (
                        "Connection error while downloading. "
                        "Retry downloading the part (attempt %s / %s)."
                    ),
                    i + 1,
                    self._max_retries,
                )
                final_exc = exc
                continue

        if final_exc is not None:
            raise MaxRetriesExceededError(final_exc)

        with open(output, "wb") as f:
            wrapped_object = CallbackIOWrapper(ctx.update, f, "write")
            downloaded_iter = response.iter_content(IO_CHUNK_SIZE)
            while True:
                final_exc = None
                for i in range(self._max_retries):
                    try:
                        part = next(downloaded_iter)
                        final_exc = None
                        break
                    except StopIteration:
                        return
                    except S3_RETRYABLE_DOWNLOAD_ERRORS as exc:
                        logger.debug(
                            (
                                "Connection error while downloading. "
                                "Retry downloading the part (attempt %s / %s)."
                            ),
                            i + 1,
                            self._max_retries,
                        )
                        final_exc = exc
                        continue

                if final_exc is not None:
                    raise MaxRetriesExceededError(final_exc)

                wrapped_object.write(part)

    def _get_content_size(self, url: str) -> int:
        """Get download content size."""
        response = requests.get(url, stream=True, timeout=DEFAULT_REQ_TIMEOUT)
        if response.status_code != 200:
            raise NotFoundError("Invalid presigned url")
        return int(response.headers["Content-Length"])


class UploadManager:
    """Upload manager."""

    def __init__(self, executor: Executor) -> None:
        """Initializes UploadManager."""
        self._executor = executor

    @classmethod
    def list_multipart_upload_objects(cls, path: Path) -> List[str]:
        """Lists file paths that requires multipart upload under the given path."""
        if path.is_file():
            paths = (
                [str(path)]
                if get_file_size(str(path)) >= S3_MULTIPART_THRESHOLD
                else []
            )
        else:
            paths = [
                str(p)
                for p in path.rglob("*")
                if p.is_file() and get_file_size(str(p)) >= S3_MULTIPART_THRESHOLD
            ]
        return paths

    @classmethod
    def list_upload_objects(cls, path: Path) -> List[str]:
        """Lists file paths that requires upload under the given path."""
        if path.is_file():
            paths = (
                [str(path)]
                if 0 < get_file_size(str(path)) < S3_MULTIPART_THRESHOLD
                else []
            )
        else:
            paths = [
                str(p)
                for p in path.rglob("*")
                if p.is_file() and 0 < get_file_size(str(p)) < S3_MULTIPART_THRESHOLD
            ]
        return paths

    def upload_file(
        self,
        upload_task: UploadTask,
        source_path: Optional[Path] = None,
    ) -> None:
        """Uploads a file in the local file system to PeriFlow."""
        if source_path is not None:
            local_path = storage_path_to_local_path(upload_task.path, source_path)
        else:
            local_path = upload_task.path

        file_size = get_file_size(local_path)

        with tqdm(
            desc=os.path.basename(local_path),
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=KiB,
        ) as pbar:
            try:
                with open(local_path, "rb") as f:
                    wrapped_object = CallbackIOWrapper(pbar.update, f, "read")
                    with Session() as s:
                        req = Request(
                            "PUT", upload_task.upload_url, data=wrapped_object
                        )
                        prep = req.prepare()
                        prep.headers["Content-Length"] = str(
                            file_size
                        )  # necessary to use ``CallbackIOWrapper``
                        response = s.send(prep)
                    if response.status_code != 200:
                        raise TransferError(
                            f"Failed to upload file ({local_path}): {response.content!r}"
                        )
            except FileNotFoundError as exc:
                raise NotFoundError(f"File '{local_path}' is not found.") from exc

    def multipart_upload_file(
        self,
        upload_task: MultipartUploadTask,
        source_path: Path,
        complete_callback: Callable[[str, str, List[Dict[str, Any]]], None],
        abort_callback: Callable[[str, str], None],
    ) -> None:
        """Uploads a file in the local file system to PeriFlow in multi-part."""
        local_path = storage_path_to_local_path(upload_task.path, source_path)
        file_size = get_file_size(local_path)
        total_num_parts = len(upload_task.upload_urls)
        uploaded_part_etags = []

        with tqdm(
            desc=os.path.basename(local_path),
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=KiB,
        ) as pbar:
            futs = {
                self._executor.submit(
                    self._upload_part,
                    file_path=local_path,
                    file_size=file_size,
                    chunk_index=idx,
                    part_number=url_info.part_number,
                    upload_url=str(url_info.upload_url),
                    ctx=pbar,
                    is_last_part=(idx == total_num_parts - 1),
                )
                for idx, url_info in enumerate(upload_task.upload_urls)
            }
            not_done = futs
            try:
                while not_done:
                    done, not_done = wait(
                        not_done, timeout=1, return_when=FIRST_EXCEPTION
                    )
                    for fut in done:
                        part_etag = fut.result()
                        uploaded_part_etags.append(part_etag.model_dump())
                complete_callback(
                    upload_task.path, upload_task.upload_id, uploaded_part_etags
                )
            except KeyboardInterrupt:
                logger.warn(
                    "Keyboard interrupted. Wait a few seconds for the shutdown."
                )
                # TODO: py38 does not support cancel_futures option. Add cancel_futures=True after deprecation.
                for fut in not_done:
                    fut.cancel()
                self._executor.shutdown(wait=False)
                abort_callback(upload_task.path, upload_task.upload_id)
                raise
            except Exception as exc:
                abort_callback(upload_task.path, upload_task.upload_id)
                raise TransferError(str(exc)) from exc

    def _upload_part(
        self,
        file_path: str,
        file_size: int,
        chunk_index: int,
        part_number: int,
        upload_url: str,
        ctx: tqdm,
        is_last_part: bool = False,
    ) -> UploadedPartETag:
        with open(file_path, "rb") as f:
            cursor = chunk_index * S3_MAX_PART_SIZE
            f.seek(cursor)

            chunk_size = min(S3_MAX_PART_SIZE, file_size - cursor)
            wrapped_object = CustomCallbackIOWrapper(ctx.update, f, "read", chunk_size)

            with Session() as s:
                req = Request("PUT", upload_url, data=wrapped_object)
                prep = req.prepare()
                prep.headers["Content-Length"] = str(chunk_size)
                response = s.send(prep)
            response.raise_for_status()

            if is_last_part:
                if f.read(S3_MAX_PART_SIZE):
                    raise TransferError(
                        "Some parts of your data is not uploaded. Please try again."
                    )

        return UploadedPartETag(etag=response.headers["ETag"], part_number=part_number)


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
