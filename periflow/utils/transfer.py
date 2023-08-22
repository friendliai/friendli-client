# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

import os
import socket
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

import requests
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from urllib3.exceptions import ReadTimeoutError

from periflow.errors import InvalidPathError, MaxRetriesExceededError, NotFoundError
from periflow.logging import logger
from periflow.utils.request import DEFAULT_REQ_TIMEOUT

KiB = 1024
MiB = KiB * KiB
GiB = MiB * KiB
IO_CHUNK_SIZE = 256 * KiB
S3_MULTIPART_THRESHOLD = 8 * MiB
S3_MAX_PART_SIZE = 8 * MiB  # 8 MiB
S3_UPLOAD_SIZE_LIMIT = 5 * GiB  # 5 GiB
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
        """Download a file without parallelism."""
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
        """Download a file in parallel."""
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
                unit_divisor=1024,
            ) as t:
                with ThreadPoolExecutor() as executor:
                    futs = [
                        executor.submit(
                            self._download_range,
                            url,
                            start,
                            start + self._max_part_size - 1,
                            f"{temp_out_prefix}.part{i}",
                            t,
                        )
                        for i, start in enumerate(chunks)
                    ]
                    not_done = futs
                    try:
                        while not_done:
                            done, not_done = wait(
                                futs, timeout=1, return_when=FIRST_EXCEPTION
                            )
                            for fut in done:
                                fut.result()
                    except KeyboardInterrupt as exc:
                        logger.warn(
                            "Keyboard interrupted. Wait a few seconds for shutting down."
                        )
                        executor.shutdown(wait=False, cancel_futures=True)
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
            iter = response.iter_content(IO_CHUNK_SIZE)
            while True:
                final_exc = None
                for i in range(self._max_retries):
                    try:
                        part = next(iter)
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
    ...
