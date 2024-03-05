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
        base_url: str,
        endpoint_id: Optional[str] = None,
        use_protobuf: bool = False,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Initializes Chat."""
        self.completions = Completions(
            base_url=base_url,
            endpoint_id=endpoint_id,
            use_protobuf=use_protobuf,
            client=client,
        )


class AsyncChat:
    """Asynchronous chat API."""

    completions: AsyncCompletions

    def __init__(
        self,
        base_url: str,
        endpoint_id: Optional[str] = None,
        use_protobuf: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initializes AsyncChat."""
        self.completions = AsyncCompletions(
            base_url=base_url,
            endpoint_id=endpoint_id,
            use_protobuf=use_protobuf,
            client=client,
        )
