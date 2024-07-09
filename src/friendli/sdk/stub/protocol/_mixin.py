# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Base implementation."""

from __future__ import annotations

import abc
import io
import json
import logging
from typing import Any, Mapping, cast

from graphql import ExecutionResult

logger = logging.getLogger(__name__)


class _RequestsLikeStubMixin:
    """Mixin for python-requests like library for stub implementation."""

    file_classes: tuple[type[Any], ...] = (io.IOBase,)

    def _prepare_request(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> Mapping[str, Any]:
        variables = variables or {}

        payload: dict[str, Any] = {"query": query}
        if operation_name:
            payload["operationName"] = operation_name

        if variables:
            payload["variables"] = variables

        nulled_variables, files = extract_files(
            variables=variables,
            file_classes=self.file_classes,
        )

        if files:
            payload["variables"] = nulled_variables
            return self._prepare_multipart_file_upload(payload, files)

        return {"json": payload}

    def _prepare_multipart_file_upload(
        self, payload: dict[str, Any], files: dict
    ) -> Mapping[str, Any]:
        """Find and extract files from the variables.

        This method inplace updates the `request_kw` dictionary.
        Extract the files present in variables and replace them by null values.

        Follows spec in https://github.com/jaydenseric/graphql-multipart-request-spec
        """
        file_map: dict[str, list[str]] = {}
        file_streams: dict[str, tuple[str, Any]] = {}

        for i, (path, val) in enumerate(files.items()):
            key = str(i)

            # Will generate something like {"0": ["variables.file"]}
            file_map[key] = [path]

            # {"0": ("variables.file", <_io.BufferedReader ...>)}
            filename = cast(str, getattr(val, "name", key))
            file_streams[key] = (filename, val)

        data: dict[str, Any] = {}

        operations_str = json.dumps(payload)
        logger.debug("operations %s", operations_str)
        data["operations"] = operations_str

        # # Add the file map field
        file_map_str = json.dumps(file_map)
        logger.debug("file_map %s", file_map_str)
        data["map"] = file_map_str

        return {"data": data, "files": file_streams}

    def _prepare_result(self, response: Mapping[str, Any]) -> ExecutionResult:
        logger.debug(">>> %s", response)

        return ExecutionResult(
            errors=response.get("errors"),
            data=response.get("data"),
            extensions=response.get("extensions"),
        )


class AsyncRequestsLikeMixin(_RequestsLikeStubMixin):
    """Async stub with python-requests library for backend."""

    @abc.abstractmethod
    async def _send_request(self, **kw: Any) -> Mapping[str, Any]:
        """Send a request to the GraphQL server."""

    async def __execute__(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> ExecutionResult:
        """Execute a GraphQL query."""
        request_kw = self._prepare_request(query, variables, operation_name)
        response = await self._send_request(**request_kw)
        return self._prepare_result(response)


class SyncRequestsLikeMixin(_RequestsLikeStubMixin):
    """Async stub with python-requests library for backend."""

    @abc.abstractmethod
    def _send_request(self, **kw: Any) -> Mapping[str, Any]:
        """Send a request to the GraphQL server."""

    def __execute__(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        operation_name: str | None = None,
    ) -> ExecutionResult:
        """Execute a GraphQL query."""
        request_kw = self._prepare_request(query, variables, operation_name)
        response = self._send_request(**request_kw)
        return self._prepare_result(response)


def extract_files(
    variables: Mapping[str, Any], file_classes: tuple[type[Any], ...]
) -> tuple[dict, dict]:
    """Find and extract files from the variables."""
    files: dict = {}

    def _recurse_extract(path: str, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, list):
            return [_recurse_extract(f"{path}.{k}", v) for k, v in enumerate(obj)]
        if isinstance(obj, dict):
            return {k: _recurse_extract(f"{path}.{k}", v) for k, v in obj.items()}
        if isinstance(obj, file_classes):
            nonlocal files
            files[path] = obj
            return None

        return obj

    nulled_variables = _recurse_extract("variables", variables)
    return nulled_variables, files
