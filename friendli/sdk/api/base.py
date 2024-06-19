# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Serving API Interface."""

# pylint: disable=no-name-in-module

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union

import grpc
import grpc._channel
import grpc.aio
import grpc.aio._call
import grpc.aio._channel
import httpx
from google.protobuf import json_format
from google.protobuf import message as pb_message
from pydantic import BaseModel
from typing_extensions import Self

from friendli.auth import get_auth_header
from friendli.errors import APIError
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


class GrpcGenerationStream(ABC, Generic[_GenerationLine]):
    """Generation stream."""

    def __init__(self, response: grpc._channel._MultiThreadedRendezvous) -> None:
        """Initializes generation stream."""
        self._iter = response

    def __iter__(self) -> Self:  # noqa: D105
        return self

    @abstractmethod
    def __next__(self) -> _GenerationLine:
        """Iterates the stream."""


class AsyncGrpcGenerationStream(ABC, Generic[_GenerationLine]):
    """Generation stream."""

    def __init__(self, response: grpc.aio._call.UnaryStreamCall) -> None:
        """Initializes generation stream."""
        self._iter = response.__aiter__()  # type: ignore

    def __aiter__(self) -> Self:  # noqa: D105
        return self

    @abstractmethod
    async def __anext__(self) -> _GenerationLine:
        """Iterates the stream."""


_HttpxClient = TypeVar("_HttpxClient", bound=Union[httpx.Client, httpx.AsyncClient])
_ProtoMsgType = TypeVar("_ProtoMsgType", bound=Type[pb_message.Message])


class BaseAPI(ABC, Generic[_HttpxClient, _ProtoMsgType]):
    """Base API interface."""

    _client: _HttpxClient

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        use_protobuf: bool = False,
    ) -> None:
        """Initializes BaseAPI."""
        self._endpoint_id = endpoint_id
        self._base_url = base_url
        self._use_protobuf = use_protobuf

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

    def _build_http_request(
        self, data: dict[str, Any], model: Optional[str] = None
    ) -> httpx.Request:
        """Build request."""
        return self._client.build_request(
            method=self._method,
            url=self._build_http_url(),
            content=self._build_content(data, model),
            files=self._build_files(data),
            headers=self._get_headers(),
            timeout=DEFAULT_REQ_TIMEOUT,
        )

    def _build_http_url(self) -> httpx.URL:
        assert self._base_url is not None
        path = ""
        if self._endpoint_id is not None:
            path = "dedicated"
        path = os.path.join(path, self._api_path)
        host = httpx.URL(self._base_url)
        return host.join(path)

    def _build_grpc_url(self) -> str:
        if self._base_url is None:
            raise ValueError(
                "You need to provide the gRPC server address through `base_url`."
            )
        return self._base_url

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

    def _build_content(
        self, data: dict[str, Any], model: Optional[str] = None
    ) -> bytes | None:
        if self._endpoint_id is not None:
            data["model"] = self._endpoint_id
        else:
            data["model"] = model

        if self._content_type.startswith("multipart/form-data"):
            return None

        if self._use_protobuf:
            pb_cls = self._request_pb_cls
            request_pb = pb_cls()
            json_format.ParseDict(data, request_pb)
            return request_pb.SerializeToString()

        return json.dumps(data).encode()

    def _build_grpc_request(
        self, data: dict[str, Any], model: Optional[str] = None
    ) -> pb_message.Message:
        if self._endpoint_id is not None:
            data["model"] = self._endpoint_id
        else:
            data["model"] = model

        pb_cls = self._request_pb_cls
        return pb_cls(**data)


class ServingAPI(BaseAPI[httpx.Client, _ProtoMsgType]):
    """Serving API interface."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        use_protobuf: bool = False,
        use_grpc: bool = False,
        client: Optional[httpx.Client] = None,
        grpc_channel: Optional[grpc.Channel] = None,
    ) -> None:
        """Initializes ServingAPI."""
        super().__init__(
            base_url=base_url,
            endpoint_id=endpoint_id,
            use_protobuf=use_protobuf,
        )

        self._use_grpc = use_grpc
        self._client = client or httpx.Client()
        self._grpc_channel = grpc_channel
        self._grpc_stub = None

    def __enter__(self) -> ServingAPI:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager with cleaning up resources."""
        self.close()

    def close(self) -> None:
        """Close the gRPC channel and HTTP client."""
        if self._grpc_channel:
            self._grpc_channel.close()
        self._client.close()

    def _get_grpc_stub(self, channel: grpc.Channel) -> Any:
        raise NotImplementedError  # pragma: no cover

    def _request(
        self, *, data: dict[str, Any], stream: bool, model: Optional[str] = None
    ) -> Any:
        # TODO: Add retry / handle timeout and etc.
        if (
            self._base_url == "https://inference.friendli.ai"
            and self._endpoint_id is None
            and model is None
        ):
            raise ValueError("`model` is required for serverless endpoints.")
        if self._endpoint_id is not None and model is not None:
            raise ValueError("`model` is not allowed for dedicated endpoints.")

        if self._use_grpc:
            grpc_request = self._build_grpc_request(data=data, model=model)
            if not self._grpc_channel:
                self._grpc_channel = grpc.insecure_channel(self._build_grpc_url())
            try:
                if not self._grpc_stub:
                    self._grpc_stub = self._get_grpc_stub(self._grpc_channel)
            except NotImplementedError as exc:
                raise ValueError("This API does not support gRPC.") from exc
            assert self._grpc_stub
            grpc_response = self._grpc_stub.Generate(grpc_request)
            return grpc_response

        http_request = self._build_http_request(data=data, model=model)
        http_response = self._client.send(request=http_request, stream=stream)
        self._check_http_error(http_response)
        return http_response

    def _check_http_error(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if response.status_code == 404:
                raise APIError(
                    "Endpoint is not found. This may be due to an invalid model name. "
                    "See https://docs.friendli.ai/guides/serverless_endpoints/pricing "
                    "to find out availble models."
                ) from exc

            resp_content = response.read()
            raise APIError(resp_content.decode()) from exc


class AsyncServingAPI(BaseAPI[httpx.AsyncClient, _ProtoMsgType]):
    """Async Serving API interface."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        use_protobuf: bool = False,
        use_grpc: bool = False,
        client: Optional[httpx.AsyncClient] = None,
        grpc_channel: Optional[grpc.aio.Channel] = None,
    ) -> None:
        """Initializes AsyncServingAPI."""
        super().__init__(
            base_url=base_url, endpoint_id=endpoint_id, use_protobuf=use_protobuf
        )

        self._use_grpc = use_grpc
        self._client = client or httpx.AsyncClient()
        self._grpc_channel = grpc_channel
        self._grpc_stub = None

    async def __aenter__(self) -> AsyncServingAPI:
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager with cleaning up resources."""
        await self.close()

    async def close(self) -> None:
        """Close the gRPC channel and HTTP client."""
        if self._grpc_channel:
            await self._grpc_channel.close(grace=None)
        await self._client.aclose()

    def _get_grpc_stub(self, channel: grpc.aio.Channel) -> Any:
        raise NotImplementedError  # pragma: no cover

    async def _request(
        self, *, data: dict[str, Any], stream: bool, model: Optional[str] = None
    ) -> Any:
        # TODO: Add retry / handle timeout and etc.
        if (
            self._base_url == "https://inference.friendli.ai"
            and self._endpoint_id is None
            and model is None
        ):
            raise ValueError("`model` is required for serverless endpoints.")
        if self._endpoint_id is not None and model is not None:
            raise ValueError("`model` is not allowed for dedicated endpoints.")

        if self._use_grpc:
            grpc_request = self._build_grpc_request(data=data, model=model)
            if not self._grpc_channel:
                self._grpc_channel = grpc.aio.insecure_channel(self._build_grpc_url())
            try:
                if not self._grpc_stub:
                    self._grpc_stub = self._get_grpc_stub(self._grpc_channel)
            except NotImplementedError as exc:
                raise ValueError("This API does not support gRPC.") from exc
            assert self._grpc_stub
            grpc_response = self._grpc_stub.Generate(
                grpc_request, timeout=DEFAULT_REQ_TIMEOUT
            )
            return grpc_response

        http_request = self._build_http_request(data=data, model=model)
        http_response = await self._client.send(request=http_request, stream=stream)
        await self._check_http_error(http_response)

        return http_response

    async def _check_http_error(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if response.status_code == 404:
                raise APIError(
                    "Endpoint is not found. This may be due to an invalid model name. "
                    "See https://docs.friendli.ai/guides/serverless_endpoints/pricing "
                    "to find out availble models."
                ) from exc

            resp_content = await response.aread()
            raise APIError(resp_content.decode()) from exc
