# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Friendli Client."""

from __future__ import annotations

from typing import Optional

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
    ):
        """Initializes Friendli."""
        super().__init__(
            token=token,
            team_id=team_id,
            project_id=project_id,
            endpoint_id=endpoint_id,
            base_url=base_url,
            use_protobuf=use_protobuf,
        )

        base_url = base_url or INFERENCE_ENDPOINT_URL
        self.completions = Completions(
            base_url=base_url, endpoint_id=self._endpoint_id, use_protobuf=use_protobuf
        )
        self.chat = Chat(
            base_url=base_url, endpoint_id=self._endpoint_id, use_protobuf=use_protobuf
        )
        self.images = Images(base_url=base_url, endpoint_id=self._endpoint_id)

        endpoint_client = EndpointGqlClient()
        model_client = ModelGqlClient()
        self.endpoint = EndpointApi(client=endpoint_client)
        self.model = ModelApi(client=model_client)


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
    ):
        """Initializes AsyncFriendli."""
        super().__init__(
            token=token,
            team_id=team_id,
            project_id=project_id,
            endpoint_id=endpoint_id,
            base_url=base_url,
            use_protobuf=use_protobuf,
        )

        base_url = base_url or INFERENCE_ENDPOINT_URL
        self.completions = AsyncCompletions(
            base_url=base_url, endpoint_id=self._endpoint_id, use_protobuf=use_protobuf
        )
        self.chat = AsyncChat(
            base_url=base_url, endpoint_id=self._endpoint_id, use_protobuf=use_protobuf
        )
        self.images = AsyncImages(base_url=base_url, endpoint_id=self._endpoint_id)
