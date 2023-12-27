# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Friendli Chat API."""

from __future__ import annotations

from typing import Optional

import httpx

from friendli.sdk.api.chat.completions import AsyncCompletions, Completions


class Chat:
    """Chat API."""

    completions: Completions

    def __init__(
        self,
        deployment_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Initializes Chat."""
        self.completions = Completions(
            deployment_id=deployment_id, endpoint=endpoint, client=client
        )


class AsyncChat:
    """Asynchronous chat API."""

    completions: AsyncCompletions

    def __init__(
        self,
        deployment_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initializes AsyncChat."""
        self.completions = AsyncCompletions(
            deployment_id=deployment_id, endpoint=endpoint, client=client
        )
