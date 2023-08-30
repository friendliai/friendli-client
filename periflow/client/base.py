# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Client Service."""

from __future__ import annotations

import copy
import math
import os
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
from urllib.parse import urljoin, urlparse

import requests
from requests.models import Response
from tqdm import tqdm

from periflow.auth import auto_token_refresh, get_auth_header
from periflow.context import get_current_group_id, get_current_project_id
from periflow.di.injector import get_injector
from periflow.errors import AuthTokenNotFoundError
from periflow.utils.format import secho_error_and_exit
from periflow.utils.fs import (
    S3_MPU_PART_MAX_SIZE,
    get_file_size,
    get_total_file_size,
    storage_path_to_local_path,
    upload_file,
    upload_part,
)
from periflow.utils.request import DEFAULT_REQ_TIMEOUT, decode_http_err
from periflow.utils.url import URLProvider

T = TypeVar("T", bound=Union[int, str, uuid.UUID])


def safe_request(
    func: Callable[..., Response], *, err_prefix: str = ""
) -> Callable[..., Response]:
    """Wrapper function to send requests with an error checking."""
    if err_prefix:
        err_prefix = err_prefix.rstrip() + "\n"

    @wraps(func)
    def wrapper(*args, **kwargs) -> Response:  # type: ignore
        try:
            return func(*args, **kwargs)
        except requests.HTTPError as exc:
            # typer.secho(exc.response.content)
            secho_error_and_exit(err_prefix + decode_http_err(exc))

    return wrapper


@dataclass
class URLTemplate:
    """URL template."""

    pattern: Template

    def render(
        self, pk: Optional[T] = None, path: Optional[str] = None, **kwargs
    ) -> str:
        """Render URLTemplate.

        Args:
            pk: Primary key of a resource.
            path: Additional URL path to attach.

        """
        if pk is None and path is None:
            return self.pattern.substitute(**kwargs)

        pattern = copy.deepcopy(self.pattern)
        need_trailing_slash = pattern.template.endswith("/")

        if pk is not None:
            pattern.template = urljoin(pattern.template + "/", str(pk))
            if need_trailing_slash:
                pattern.template += "/"

        if path is not None:
            pattern.template = urljoin(pattern.template + "/", path.rstrip("/"))
            if need_trailing_slash:
                pattern.template += "/"

        return pattern.substitute(**kwargs)

    def get_base_url(self) -> str:
        """Get a base URL."""
        result = urlparse(self.pattern.template)
        return f"{result.scheme}://{result.hostname}"

    def attach_pattern(self, pattern: str) -> None:
        """Attach a URL path pattern."""
        self.pattern.template = urljoin(self.pattern.template + "/", pattern)

    def replace_path(self, path: str):
        """Replace a URL path."""
        result = urlparse(self.pattern.template)
        result = result._replace(path=path)
        self.pattern.template = result.geturl()

    def copy(self) -> "URLTemplate":
        """Get a copy of this URL template."""
        return URLTemplate(pattern=Template(self.pattern.template))


class Client(ABC, Generic[T]):
    """Base interface of client to PeriFlow system."""

    def __init__(self, **kwargs):
        """Initialize client."""
        injector = get_injector()
        self.url_provider = injector.get(URLProvider)
        self.url_template = URLTemplate(self.url_path)
        self.url_kwargs = kwargs

    @property
    @abstractmethod
    def url_path(self) -> Template:
        """URL path template to render."""

    @property
    def default_request_options(self) -> Dict[str, Any]:
        """Common request options."""
        try:
            return {
                "headers": get_auth_header(),
                "timeout": DEFAULT_REQ_TIMEOUT,
            }
        except AuthTokenNotFoundError as exc:
            secho_error_and_exit(str(exc))

    @auto_token_refresh
    def list(self, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to list objects."""
        return requests.get(
            self.url_template.render(path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )

    @auto_token_refresh
    def retrieve(self, pk: T, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to retrieve a specific object."""
        return requests.get(
            self.url_template.render(pk=pk, path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )

    @auto_token_refresh
    def post(self, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to post an object."""
        return requests.post(
            self.url_template.render(path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )

    @auto_token_refresh
    def partial_update(self, pk: T, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to partially update a specific object."""
        return requests.patch(
            self.url_template.render(pk=pk, path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )

    @auto_token_refresh
    def delete(self, pk: T, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to delete a specific obejct."""
        return requests.delete(
            self.url_template.render(pk=pk, path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )

    @auto_token_refresh
    def update(self, pk: T, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to update a specific object."""
        return requests.put(
            self.url_template.render(pk=pk, path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )

    def bare_post(self, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request without automatic auth."""
        r = requests.post(
            self.url_template.render(path=path, **self.url_kwargs),
            timeout=DEFAULT_REQ_TIMEOUT,
            **kwargs,
        )
        r.raise_for_status()
        return r


class UserRequestMixin:
    """Mixin class of user-level requests."""

    user_id: uuid.UUID

    @auto_token_refresh
    def _userinfo(self) -> Response:
        injector = get_injector()
        url_provider = injector.get(URLProvider)
        try:
            return requests.get(
                url_provider.get_auth_uri("oauth2/userinfo"),
                headers=get_auth_header(),
                timeout=DEFAULT_REQ_TIMEOUT,
            )
        except AuthTokenNotFoundError as exc:
            secho_error_and_exit(str(exc))

    def get_current_userinfo(self) -> Dict[str, Any]:
        """Get the currently logged-in user info."""
        response = safe_request(self._userinfo, err_prefix="Failed to get userinfo.")()
        return response.json()

    def get_current_user_id(self) -> uuid.UUID:
        """Get the currently logged-in user ID."""
        userinfo = self.get_current_userinfo()
        return uuid.UUID(userinfo["sub"].split("|")[1])

    def initialize_user(self):
        """Initialize user settings."""
        self.user_id = self.get_current_user_id()


class GroupRequestMixin:
    """Mixin class of organization-level requests."""

    group_id: uuid.UUID

    def initialize_group(self):
        """Initialize organization settings."""
        group_id = get_current_group_id()
        if group_id is None:
            secho_error_and_exit("Organization is not set.")
        self.group_id = group_id  # type: ignore


class ProjectRequestMixin:
    """Mixin class of project-level requests."""

    project_id: uuid.UUID

    def initialize_project(self):
        """Initialize project settings."""
        project_id = get_current_project_id()
        if project_id is None:
            secho_error_and_exit("Project is not set.")
        self.project_id = project_id  # type: ignore


class UploadableClient(Client[T], Generic[T]):
    """Uploadable client."""

    def get_spu_urls(
        self,
        obj_id: T,
        storage_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """Get single part upload URLs for multiple files.

        Args:
            obj_id (T): Uploadable object ID
            storage_paths (List[str]): A list of cloud storage paths of target files

        Returns:
            List[Dict[str, Any]]: A response body that has presigned URL info to upload files.

        """
        response = safe_request(self.post, err_prefix="Failed to get presigned URLs.")(
            path=f"{obj_id}/upload/", json={"paths": storage_paths}
        )
        return response.json()

    def get_mpu_urls(
        self,
        obj_id: T,
        local_paths: List[str],
        storage_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """Get multipart upload URLs for multiple file-like objects.

        Args:
            obj_id (T): Uploadable object ID
            local_paths (List[str]): A list local paths to target files. The path can be
                either absolute or relative.
            storage_paths (List[str]): A list of storage paths to target files.

        Returns:
            List[Dict[str, Any]]: A list of multipart upload responses for multiple target files.

        """
        start_mpu_resps = []
        for local_path, storage_path in zip(local_paths, storage_paths):
            num_parts = math.ceil(get_file_size(local_path) / S3_MPU_PART_MAX_SIZE)
            response = safe_request(
                self.post,
                err_prefix="Failed to get presigned URLs for multipart upload.",
            )(
                path=f"{obj_id}/start_mpu/",
                json={
                    "path": storage_path,
                    "num_parts": num_parts,
                },
            )
            start_mpu_resps.append(response.json())
        return start_mpu_resps

    def complete_mpu(
        self, obj_id: T, path: str, upload_id: str, parts: List[Dict[str, Any]]
    ) -> None:
        """Complete multipart upload.

        Args:
            obj_id (T): Uploadable object ID
            path (str): Path to the uploaded file
            upload_id (str): Upload ID
            parts (List[Dict[str, Any]]): A list of upload part info

        """
        safe_request(
            self.post, err_prefix=f"Failed to complete multipart upload for {path}"
        )(
            path=f"{obj_id}/complete_mpu/",
            json={
                "path": path,
                "upload_id": upload_id,
                "parts": parts,
            },
        )

    def abort_mpu(self, obj_id: T, path: str, upload_id: str) -> None:
        """Abort multipart upload.

        Args:
            obj_id (T): Uploadable object ID
            path (str): Path to the target file
            upload_id (str): Upload ID
            parts (List[Dict[str, Any]]): A list of upload part info

        """
        safe_request(
            self.post, err_prefix=f"Failed to abort multipart upload for {path}"
        )(
            path=f"{obj_id}/abort_mpu/",
            json={
                "path": path,
                "upload_id": upload_id,
            },
        )

    def _multipart_upload_file(
        self,
        obj_id: T,
        file_path: str,
        url_dict: Dict[str, Any],
        ctx: tqdm,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Start multi-part uploading for a file."""
        parts = []
        upload_id = url_dict["upload_id"]
        object_path = url_dict["path"]
        upload_urls = url_dict["upload_urls"]
        total_num_parts = len(upload_urls)
        try:
            futs = [
                executor.submit(
                    upload_part,
                    file_path=file_path,
                    chunk_index=idx,
                    part_number=url_info["part_number"],
                    upload_url=url_info["upload_url"],
                    ctx=ctx,
                    is_last_part=(idx == total_num_parts - 1),
                )
                for idx, url_info in enumerate(upload_urls)
            ]
            wait(futs, return_when=FIRST_EXCEPTION)
            for fut in futs:
                exc = fut.exception()
                if exc is not None:
                    raise exc
                parts.append(fut.result())
            self.complete_mpu(obj_id, object_path, upload_id, parts)
        except KeyboardInterrupt:
            secho_error_and_exit("File upload is aborted.")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.abort_mpu(obj_id, object_path, upload_id)
            secho_error_and_exit(f"File upload is aborted: ({exc!r})")

    def upload_files(
        self,
        obj_id: T,
        spu_url_dicts: List[Dict[str, str]],
        mpu_url_dicts: List[Dict[str, Any]],
        source_path: Path,
        max_workers: int = min(
            32, (os.cpu_count() or 1) + 4
        ),  # default of ``ThreadPoolExecutor``
    ) -> None:
        """Upload files.

        Args:
            obj_id (T): PeriFlow resource(i.e., checkpoint) object ID.
            spu_url_dicts (List[Dict[str, str]]): Single-part upload URL info.
            mpu_url_dicts (List[Dict[str, Any]]): Multi-part upload URL info.
            source_path (Path): Path to the local source directory.
            max_workers (int, optional): Max number of workers for the concurrent uploading.

        """
        spu_local_paths = [
            storage_path_to_local_path(url_info["path"], source_path)
            for url_info in spu_url_dicts
        ]
        mpu_local_paths = [
            storage_path_to_local_path(url_info["path"], source_path)
            for url_info in mpu_url_dicts
        ]
        total_size = get_total_file_size(spu_local_paths + mpu_local_paths)
        spu_urls = [url_info["upload_url"] for url_info in spu_url_dicts]

        with tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            # NOTE: excessive concurrency may results in "No buffer space available" error.
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Normal upload for files with size < 5 GiB
                futs = [
                    executor.submit(upload_file, local_path, upload_url, pbar)
                    for (local_path, upload_url) in zip(spu_local_paths, spu_urls)
                ]
                # Multipart upload for large files with sizes >= 5 GiB
                futs.extend(
                    [
                        executor.submit(
                            self._multipart_upload_file,
                            obj_id=obj_id,
                            file_path=local_path,
                            url_dict=url_dict,
                            ctx=pbar,
                            executor=executor,
                        )
                        for (local_path, url_dict) in zip(
                            mpu_local_paths, mpu_url_dicts
                        )
                    ]
                )
                wait(futs, return_when=FIRST_EXCEPTION)
                for fut in futs:
                    exc = fut.exception()
                    if exc is not None:
                        raise exc
