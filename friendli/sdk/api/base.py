# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Serving API Interface."""

# pylint: disable=no-name-in-module

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union

import httpx
from google.protobuf import json_format
from pydantic import BaseModel
from typing_extensions import Self

from friendli.auth import get_auth_header
from friendli.errors import APIError, InvalidConfigError
from friendli.schema.api.v1.codegen.chat_completions_pb2 import V1ChatCompletionsRequest
from friendli.schema.api.v1.codegen.completions_pb2 import V1CompletionsRequest
from friendli.schema.api.v1.codegen.text_to_image_pb2 import V1TextToImageRequest
from friendli.utils.request import DEFAULT_REQ_TIMEOUT

_GenerationLine = TypeVar("_GenerationLine", bound=BaseModel)


class GenerationStream(ABC, Generic[_GenerationLine]):
    """Generation stream."""

    def __init__(self, response: httpx.Response) -> None:
        """Initializes generation stream."""
        self._iter = response.iter_lines()

    def __iter__(self) -> Self:  # noqa: D105
        return self

    @abstractmethod
    def __next__(self) -> _GenerationLine:
        """Iterates the stream."""


class AsyncGenerationStream(ABC, Generic[_GenerationLine]):
    """Asynchronous generation stream."""

    def __init__(self, response: httpx.Response) -> None:
        """Initializes generation stream."""
        self._iter = response.aiter_lines()

    def __aiter__(self) -> Self:  # noqa: D105
        return self

    @abstractmethod
    async def __anext__(self) -> _GenerationLine:
        """Iterates the stream."""


_HttpxClient = TypeVar("_HttpxClient", bound=Union[httpx.Client, httpx.AsyncClient])
_ProtoMsgType = TypeVar(
    "_ProtoMsgType",
    bound=Union[
        Type[V1CompletionsRequest],
        Type[V1ChatCompletionsRequest],
        Type[V1TextToImageRequest],
    ],
)


class BaseAPI(ABC, Generic[_HttpxClient, _ProtoMsgType]):
    """Base API interface."""

    _client: _HttpxClient

    def __init__(
        self,
        deployment_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        """Initializes BaseAPI."""
        self._deployment_id = deployment_id

        if deployment_id is None and endpoint is None:
            raise InvalidConfigError(
                "One of 'deployment_id' and 'endpoint' should be provided."
            )
        if deployment_id is not None and endpoint is not None:
            raise InvalidConfigError(
                "Only provide one between 'deployment_id' and 'endpoint'."
            )

        # if deployment_id is not None:
        #     client = DeploymentGqlClient()
        #     deployment = client.get_deployment(deployment_id)
        #     endpoint = deployment["endpoint"]
        #     if not endpoint:
        #         raise NotFoundError("Active endpoint for the deployment is not found.")
        #     self._host = httpx.URL(get_host(endpoint))
        # elif endpoint is not None:
        #     self._host = httpx.URL(endpoint)
        if endpoint is not None:
            self._host = httpx.URL(endpoint)
        else:
            raise ValueError("endpoint is not configured.")

    @property
    @abstractmethod
    def _api_path(self) -> str:
        """API URL path."""

    @property
    @abstractmethod
    def _method(self) -> str:
        """API call method."""

    @property
    @abstractmethod
    def _content_type(self) -> str:
        """Request content type."""

    @property
    @abstractmethod
    def _request_pb_cls(self) -> _ProtoMsgType:
        """Protobuf message class to serialize the data of request body."""

    def _build_request(
        self, data: dict[str, Any], model: Optional[str] = None
    ) -> httpx.Request:
        """Build request."""
        if self._deployment_id is None and model is None:
            raise ValueError("`model` is required for serverless endpoints.")
        if self._deployment_id is not None and model is not None:
            raise ValueError("`model` is not allowed for dedicated endpoints.")

        return self._client.build_request(
            method=self._method,
            url=self._build_url(model),
            content=self._build_content(data),
            files=self._build_files(data),
            headers=self._get_headers(),
            timeout=DEFAULT_REQ_TIMEOUT,
        )

    def _build_url(self, model: Optional[str] = None) -> httpx.URL:
        path = ""
        if model is not None:
            path = model
        if self._deployment_id is not None:
            path = self._deployment_id
        path = os.path.join(path, self._api_path)
        return self._host.join(path)

    def _get_headers(self) -> Dict[str, Any]:
        return {
            "Content-Type": self._content_type,
            **get_auth_header(),
        }

    def _build_files(self, data: dict[str, Any]) -> dict[str, Any] | None:
        if self._content_type.startswith("multipart/form-data"):
            files = {}
            for key, val in data.items():
                if val is not None:
                    files[key] = (None, val)
            return files
        return None

    def _build_content(self, data: dict[str, Any]) -> bytes | None:
        if self._content_type.startswith("multipart/form-data"):
            return None

        if self._deployment_id is None:
            return json.dumps(data).encode()

        pb_cls = self._request_pb_cls
        request_pb = pb_cls()
        json_format.ParseDict(data, request_pb)
        return request_pb.SerializeToString()


class ServingAPI(BaseAPI[httpx.Client, _ProtoMsgType]):
    """Serving API interface."""

    def __init__(
        self,
        deployment_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Initializes ServingAPI."""
        super().__init__(deployment_id=deployment_id, endpoint=endpoint)
        self._client = client or httpx.Client()

    def _request(
        self, *, data: dict[str, Any], stream: bool, model: Optional[str] = None
    ) -> httpx.Response:
        # TODO: Add retry / handle timeout and etc.
        request = self._build_request(data=data, model=model)
        response = self._client.send(request=request, stream=stream)
        self._check_http_error(response)

        return response

    def _check_http_error(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if response.status_code == 404:
                raise APIError(
                    "Endpoint is not found. This may be due to an invalid model name. "
                    "See https://docs.periflow.ai/guides/serverless_endpoints/pricing "
                    "to find out availble models."
                ) from exc

            resp_content = response.read()
            raise APIError(resp_content.decode()) from exc


class AsyncServingAPI(BaseAPI[httpx.AsyncClient, _ProtoMsgType]):
    """Async Serving API interface."""

    def __init__(
        self,
        deployment_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initializes AsyncServingAPI."""
        super().__init__(deployment_id=deployment_id, endpoint=endpoint)
        self._client = client or httpx.AsyncClient()

    async def _request(
        self, *, data: dict[str, Any], stream: bool, model: Optional[str] = None
    ) -> httpx.Response:
        # TODO: Add retry / handle timeout and etc.
        request = self._build_request(data=data, model=model)
        response = await self._client.send(request=request, stream=stream)
        await self._check_http_error(response)

        return response

    async def _check_http_error(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if response.status_code == 404:
                raise APIError(
                    "Endpoint is not found. This may be due to an invalid model name. "
                    "See https://docs.periflow.ai/guides/serverless_endpoints/pricing "
                    "to find out availble models."
                ) from exc

            resp_content = await response.aread()
            raise APIError(resp_content.decode()) from exc
