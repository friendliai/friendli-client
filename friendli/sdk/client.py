# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Friendli Client."""

from __future__ import annotations

from typing import Optional

import friendli
from friendli.client.project import ProjectClient, find_project_id
from friendli.client.user import UserGroupClient, UserGroupProjectClient
from friendli.errors import AuthorizationError
from friendli.sdk.api.chat.chat import AsyncChat, Chat
from friendli.sdk.api.completions import AsyncCompletions, Completions
from friendli.sdk.resource.catalog import Catalog
from friendli.sdk.resource.checkpoint import Checkpoint
from friendli.sdk.resource.credential import Credential
from friendli.sdk.resource.deployment import Deployment

SERVERLESS_ENDPOINT_URL = "https://inference.friendli.ai"


class FriendliClientBase:
    """Base class of Friendli client."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
    ):
        """Initializes FriendliClientBase."""
        if api_key is not None:
            friendli.api_key = api_key

        if project is not None:
            user_group_client = UserGroupClient()
            try:
                org = user_group_client.get_group_info()
            except IndexError as exc:
                raise AuthorizationError(
                    "Does have not permission to any organization."
                ) from exc
            friendli.org_id = org["id"]

            user_group_project_client = UserGroupProjectClient()
            project_id = find_project_id(
                projects=user_group_project_client.list_projects(),
                project_name=project,
            )

            project_client = ProjectClient()
            if project_client.check_project_membership(pf_project_id=project_id):
                friendli.project_id = str(project_id)
            else:
                raise AuthorizationError(
                    f"Does not have permission to the project '{project}'."
                )


class FriendliResource(FriendliClientBase):
    """Friendli resource client."""

    deployment: Deployment
    checkpoint: Checkpoint
    catalog: Catalog
    credential: Credential

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
    ):
        """Initializes FriendliResource."""
        super().__init__(
            api_key=api_key,
            project=project,
        )

        self.deployment = Deployment()
        self.checkpoint = Checkpoint()
        self.catalog = Catalog()
        self.credential = Credential()


class Friendli(FriendliClientBase):
    """Friendli API client."""

    completions: Completions
    chat: Chat

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ):
        """Initializes Friendli."""
        super().__init__(
            api_key=api_key,
            project=project,
        )

        endpoint = SERVERLESS_ENDPOINT_URL if deployment_id is None else None
        self.completions = Completions(deployment_id=deployment_id, endpoint=endpoint)
        self.chat = Chat(deployment_id=deployment_id, endpoint=endpoint)


class AsyncFriendli(FriendliClientBase):
    """Async Friendli API client."""

    completions: AsyncCompletions
    chat: AsyncChat

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ):
        """Initializes AsyncFriendli."""
        super().__init__(
            api_key=api_key,
            project=project,
        )

        endpoint = SERVERLESS_ENDPOINT_URL if deployment_id is None else None
        self.completions = AsyncCompletions(
            deployment_id=deployment_id, endpoint=endpoint
        )
        self.chat = AsyncChat(deployment_id=deployment_id, endpoint=endpoint)
