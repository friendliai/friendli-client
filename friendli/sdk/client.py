# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Friendli Client."""

from __future__ import annotations

from typing import Optional

import friendli
from friendli.client.graphql.base import get_default_gql_client
from friendli.client.graphql.deployment import DeploymentGqlClient
from friendli.sdk.api.chat.chat import AsyncChat, Chat
from friendli.sdk.api.completions import AsyncCompletions, Completions
from friendli.sdk.api.images.images import AsyncImages, Images
from friendli.sdk.resource.deployment import DeploymentAPI

SERVERLESS_ENDPOINT_URL = "https://inference.friendli.ai"


class FriendliClientBase:
    """Base class of Friendli client."""

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Initializes FriendliClientBase."""
        if token is not None:
            friendli.token = token
        if team_id is not None:
            friendli.team_id = team_id
        if project_id is not None:
            friendli.project_id = project_id


class Friendli(FriendliClientBase):
    """Friendli API client."""

    completions: Completions
    chat: Chat
    images: Images
    deployment: DeploymentAPI

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Initializes Friendli."""
        super().__init__(token=token, team_id=team_id, project_id=project_id)

        endpoint = SERVERLESS_ENDPOINT_URL
        self.completions = Completions(endpoint=endpoint)
        self.chat = Chat(endpoint=endpoint)
        self.images = Images(endpoint=endpoint)

        gql_client = get_default_gql_client()
        deployment_client = DeploymentGqlClient(client=gql_client)
        self.deployment = DeploymentAPI(client=deployment_client)


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
    ):
        """Initializes AsyncFriendli."""
        super().__init__(token=token, team_id=team_id)

        endpoint = SERVERLESS_ENDPOINT_URL
        self.completions = AsyncCompletions(endpoint=endpoint)
        self.chat = AsyncChat(endpoint=endpoint)
        self.images = AsyncImages(endpoint=endpoint)
