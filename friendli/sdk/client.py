# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Friendli Client."""

from __future__ import annotations

from typing import Optional, Union

import grpc
import grpc.aio

import friendli
from friendli.client.graphql.endpoint import EndpointGqlClient
from friendli.client.graphql.model import ModelGqlClient
from friendli.sdk.api.chat.chat import AsyncChat, Chat
from friendli.sdk.api.completions import AsyncCompletions, Completions
from friendli.sdk.api.images.images import AsyncImages, Images
from friendli.sdk.resource.endpoint import EndpointApi
from friendli.sdk.resource.model import ModelApi

INFERENCE_ENDPOINT_URL = "https://inference.friendli.ai"


class FriendliClientBase:
    """Base class of Friendli client."""

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        base_url: Optional[str] = None,
        use_protobuf: bool = False,
        use_grpc: bool = False,
        grpc_channel: Optional[Union[grpc.Channel, grpc.aio.Channel]] = None,
    ):
        """Initializes FriendliClientBase."""
        if token is not None:
            friendli.token = token
        if team_id is not None:
            friendli.team_id = team_id
        if project_id is not None:
            friendli.project_id = project_id
        self._endpoint_id = endpoint_id
        self._base_url = base_url
        self._use_protobuf = use_protobuf

        if use_grpc:
            if base_url is None and grpc_channel is None:
                raise ValueError(
                    "One of `base_url` and `grpc_channel` should be set when `use_grpc=True`."
                )
        else:
            if grpc_channel is not None:
                raise ValueError(
                    "Setting `use_grpc=True` is required when `grpc_channel` is set."
                )
            if base_url is None:
                self._base_url = INFERENCE_ENDPOINT_URL


class Friendli(FriendliClientBase):
    """Friendli API client."""

    completions: Completions
    chat: Chat
    images: Images
    endpoint: EndpointApi
    model: ModelApi

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        base_url: Optional[str] = None,
        use_protobuf: bool = False,
        use_grpc: bool = False,
        grpc_channel: Optional[grpc.Channel] = None,
    ):
        """Initializes Friendli."""
        super().__init__(
            token=token,
            team_id=team_id,
            project_id=project_id,
            endpoint_id=endpoint_id,
            base_url=base_url,
            use_protobuf=use_protobuf,
            use_grpc=use_grpc,
            grpc_channel=grpc_channel,
        )

        self.completions = Completions(
            base_url=self._base_url,
            endpoint_id=self._endpoint_id,
            use_protobuf=use_protobuf,
            use_grpc=use_grpc,
            grpc_channel=grpc_channel,
        )
        self.chat = Chat(
            base_url=self._base_url,
            endpoint_id=self._endpoint_id,
            use_protobuf=use_protobuf,
            use_grpc=use_grpc,
            grpc_channel=grpc_channel,
        )
        self.images = Images(base_url=self._base_url, endpoint_id=self._endpoint_id)

        endpoint_client = EndpointGqlClient()
        model_client = ModelGqlClient()
        self.endpoint = EndpointApi(client=endpoint_client)
        self.model = ModelApi(client=model_client)

    def __enter__(self) -> Friendli:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and close resources."""
        self.close()

    def close(self) -> None:
        """Clean up all clients' resources."""
        self.completions.close()
        self.chat.close()
        self.images.close()


class AsyncFriendli(FriendliClientBase):
    """Async Friendli API client."""

    completions: AsyncCompletions
    chat: AsyncChat
    images: AsyncImages

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        base_url: Optional[str] = None,
        use_protobuf: bool = False,
        use_grpc: bool = False,
        grpc_channel: Optional[grpc.aio.Channel] = None,
    ):
        """Initializes AsyncFriendli."""
        super().__init__(
            token=token,
            team_id=team_id,
            project_id=project_id,
            endpoint_id=endpoint_id,
            base_url=base_url,
            use_protobuf=use_protobuf,
            use_grpc=use_grpc,
            grpc_channel=grpc_channel,
        )

        self.completions = AsyncCompletions(
            base_url=self._base_url,
            endpoint_id=self._endpoint_id,
            use_protobuf=use_protobuf,
            use_grpc=use_grpc,
            grpc_channel=grpc_channel,
        )
        self.chat = AsyncChat(
            base_url=self._base_url,
            endpoint_id=self._endpoint_id,
            use_protobuf=use_protobuf,
            use_grpc=use_grpc,
            grpc_channel=grpc_channel,
        )
        self.images = AsyncImages(
            base_url=self._base_url, endpoint_id=self._endpoint_id
        )

    async def __aenter__(self) -> AsyncFriendli:
        """Enter the asynchronous context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the asynchronous context manager and close resources."""
        await self.close()

    async def close(self) -> None:
        """Clean up all clients' resources."""
        await self.completions.close()
        await self.chat.close()
        await self.images.close()
