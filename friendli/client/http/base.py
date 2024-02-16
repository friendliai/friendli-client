# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Client Service."""

from __future__ import annotations

import copy
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
from urllib.parse import urljoin, urlparse

import requests
from requests import HTTPError
from requests.models import Response
from urllib3.exceptions import ConnectTimeoutError, ReadTimeoutError

from friendli.auth import get_auth_header, safe_request
from friendli.context import get_current_project_id, get_current_team_id
from friendli.di.injector import get_injector
from friendli.errors import APIError, MaxRetriesExceededError
from friendli.logging import logger
from friendli.utils.format import secho_error_and_exit
from friendli.utils.fs import get_file_size
from friendli.utils.request import DEFAULT_REQ_TIMEOUT
from friendli.utils.transfer import S3_MULTIPART_CHUNK_SIZE, ChunksizeAdjuster
from friendli.utils.url import URLProvider

T = TypeVar("T", bound=Union[int, str, uuid.UUID])

RETRYABLE_REQUEST_ERRORS = (
    ConnectionError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    ReadTimeoutError,
    ConnectTimeoutError,
)
API_REQUEST_MAX_RETRIES = 3
DEFAULT_PAGINATION_SIZE = 20


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


class RequestInterface:
    """Request API mixin."""

    def check_request(self, func: Callable[..., Response]) -> Callable[..., Response]:
        """Wrapper function to send requests with the retry and error handling."""

        @wraps(func)
        def wrapper(*args, **kwargs) -> Response:  # type: ignore
            final_exc = None
            for i in range(API_REQUEST_MAX_RETRIES):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_REQUEST_ERRORS:
                    logger.info(
                        ("Retry the failed request (attempt %s / %s)."),
                        i + 1,
                        API_REQUEST_MAX_RETRIES,
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    final_exc = exc
            raise MaxRetriesExceededError(final_exc)

        return wrapper

    def paginated_get(
        self,
        callback: Callable[..., Response],
        path: Optional[str] = None,
        limit: int = DEFAULT_PAGINATION_SIZE,
        params: Optional[dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List objects with pagination."""
        page_size = min(DEFAULT_PAGINATION_SIZE, limit)
        params = {"limit": page_size, **(params or {})}
        response_dict = callback(path=path, params=params).json()
        items = response_dict["results"]
        next_cursor = response_dict["next_cursor"]

        while next_cursor is not None and len(items) < limit:
            response_dict = callback(
                path=path, params={**params, "cursor": next_cursor}
            ).json()
            items.extend(response_dict["results"])
            next_cursor = response_dict["next_cursor"]

        return items


class HttpClient(ABC, Generic[T], RequestInterface):
    """Base interface of client to Friendli system."""

    def __init__(self, **kwargs):
        """Initialize client."""
        self.injector = get_injector()
        self.url_provider = self.injector.get(URLProvider)
        self.url_template = URLTemplate(Template(self.url_path))
        self.url_kwargs = kwargs

    @property
    @abstractmethod
    def url_path(self) -> str:
        """URL path template to render."""

    @property
    def default_request_options(self) -> Dict[str, Any]:
        """Common request options."""
        return {
            "headers": get_auth_header(),
            "timeout": DEFAULT_REQ_TIMEOUT,
        }

    def list(
        self,
        path: Optional[str] = None,
        *,
        pagination: bool = True,
        limit: int = DEFAULT_PAGINATION_SIZE,
        **kwargs,
    ) -> Any:
        """Send a request to list objects."""
        if pagination:
            page_size = min(DEFAULT_PAGINATION_SIZE, limit)
            params = kwargs.pop("params", {})
            params = {"limit": page_size, "page_size": page_size, **params}
            data = self._list(path=path, params=params, **kwargs)
            items = data["results"]
            next_cursor = data["next_cursor"]

            while next_cursor is not None and len(items) < limit:
                params["limit"] = params["page_size"] = min(
                    page_size, limit - len(items)
                )
                data = self._list(
                    path=path,
                    params={**params, "cursor": next_cursor},
                    **kwargs,
                )
                items.extend(data["results"])
                next_cursor = data["next_cursor"]

            return items
        return self._list(path=path, **kwargs)

    @safe_request
    def retrieve(self, pk: T, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to retrieve a specific object."""
        resp = self.check_request(requests.get)(
            self.url_template.render(pk=pk, path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )
        return resp

    @safe_request
    def post(self, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to post an object."""
        resp = self.check_request(requests.post)(
            self.url_template.render(path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )
        return resp

    @safe_request
    def partial_update(self, pk: T, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to partially update a specific object."""
        resp = self.check_request(requests.patch)(
            self.url_template.render(pk=pk, path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )
        return resp

    @safe_request
    def delete(self, pk: T, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to delete a specific obejct."""
        resp = self.check_request(requests.delete)(
            self.url_template.render(pk=pk, path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )
        return resp

    @safe_request
    def update(self, pk: T, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to update a specific object."""
        resp = self.check_request(requests.put)(
            self.url_template.render(pk=pk, path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )
        return resp

    def bare_post(self, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request without automatic auth."""
        resp = self.check_request(requests.post)(
            self.url_template.render(path=path, **self.url_kwargs),
            timeout=DEFAULT_REQ_TIMEOUT,
            **kwargs,
        )
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            raise APIError(str(exc)) from exc
        return resp

    @safe_request
    def _list(self, path: Optional[str] = None, **kwargs) -> Response:
        """Send a request to list objects."""
        resp = self.check_request(requests.get)(
            self.url_template.render(path=path, **self.url_kwargs),
            **self.default_request_options,
            **kwargs,
        )
        return resp


class UserRequestMixin(RequestInterface):
    """Mixin class of user-level requests."""

    user_id: uuid.UUID

    @safe_request
    def get_current_user_info(self) -> Response:
        """Gets the current user info."""
        injector = get_injector()
        url_provider = injector.get(URLProvider)
        return self.check_request(requests.get)(
            url_provider.get_auth_uri("oauth2/userinfo"),
            headers=get_auth_header(),
            timeout=DEFAULT_REQ_TIMEOUT,
        )

    def get_current_user_id(self) -> uuid.UUID:
        """Gets the currently logged-in user ID."""
        userinfo = self.get_current_user_info()
        return uuid.UUID(userinfo["sub"].split("|")[1])

    def initialize_user(self):
        """Initializes user settings."""
        self.user_id = self.get_current_user_id()


class GroupRequestMixin:
    """Mixin class of organization-level requests."""

    group_id: uuid.UUID

    def initialize_group(self):
        """Initialize organization settings."""
        group_id = get_current_team_id()
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


class UploadableClient(HttpClient[T], Generic[T]):
    """Uploadable client."""

    def get_upload_urls(
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
        data = self.post(path=f"{obj_id}/upload/", json={"paths": storage_paths})
        return data

    def get_multipart_upload_urls(
        self,
        obj_id: T,
        local_paths: List[Path],
        storage_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """Get multipart upload URLs for multiple file-like objects.

        Args:
            obj_id (T): Uploadable object ID
            local_paths (List[Path]): A list local paths to target files. The path can
                be either absolute or relative.
            storage_paths (List[str]): A list of storage paths to target files.

        Returns:
            List[Dict[str, Any]]: A list of multipart upload responses for multiple target files.

        """
        start_mpu_resps = []
        adjuster = ChunksizeAdjuster()
        for local_path, storage_path in zip(local_paths, storage_paths):
            file_size = get_file_size(local_path)
            part_size = adjuster.adjust_chunksize(
                current_chunksize=S3_MULTIPART_CHUNK_SIZE, file_size=file_size
            )
            num_parts = math.ceil(file_size / part_size)
            data = self.post(
                path=f"{obj_id}/start_mpu/",
                json={
                    "path": storage_path,
                    "num_parts": num_parts,
                },
            )
            start_mpu_resps.append(data)
        return start_mpu_resps

    def complete_multipart_upload(
        self, obj_id: T, path: str, upload_id: str, parts: List[Dict[str, Any]]
    ) -> None:
        """Complete multipart upload.

        Args:
            obj_id (T): Uploadable object ID
            path (str): Path to the uploaded file
            upload_id (str): Upload ID
            parts (List[Dict[str, Any]]): A list of upload part info

        """
        sorted_parts = sorted(parts, key=lambda part: part["part_number"])
        self.post(
            path=f"{obj_id}/complete_mpu/",
            json={
                "path": path,
                "upload_id": upload_id,
                "parts": sorted_parts,
            },
        )

    def abort_multipart_upload(self, obj_id: T, path: str, upload_id: str) -> None:
        """Abort multipart upload.

        Args:
            obj_id (T): Uploadable object ID
            path (str): Path to the target file
            upload_id (str): Upload ID
            parts (List[Dict[str, Any]]): A list of upload part info

        """
        self.post(
            path=f"{obj_id}/abort_mpu/",
            json={
                "path": path,
                "upload_id": upload_id,
            },
        )
