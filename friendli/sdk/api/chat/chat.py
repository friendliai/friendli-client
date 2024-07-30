# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Friendli Chat API."""

from __future__ import annotations

from typing import Optional

import grpc
import grpc.aio
import httpx

from friendli.sdk.api.chat.completions import AsyncCompletions, Completions


class Chat:
    """Chat API."""

    completions: Completions

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        use_protobuf: bool = False,
        use_grpc: bool = False,
        http_client: Optional[httpx.Client] = None,
        grpc_channel: Optional[grpc.Channel] = None,
    ) -> None:
        """Initializes Chat."""
        self.completions = Completions(
            base_url=base_url,
            endpoint_id=endpoint_id,
            use_protobuf=use_protobuf,
            use_grpc=use_grpc,
            http_client=http_client,
            grpc_channel=grpc_channel,
        )

    def close(self) -> None:
        """Clean up all clients' resources."""
        self.completions.close()


class AsyncChat:
    """Asynchronous chat API."""

    completions: AsyncCompletions

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        use_protobuf: bool = False,
        use_grpc: bool = False,
        http_client: Optional[httpx.AsyncClient] = None,
        grpc_channel: Optional[grpc.aio.Channel] = None,
    ) -> None:
        """Initializes AsyncChat."""
        self.completions = AsyncCompletions(
            base_url=base_url,
            endpoint_id=endpoint_id,
            use_protobuf=use_protobuf,
            use_grpc=use_grpc,
            http_client=http_client,
            grpc_channel=grpc_channel,
        )

    async def close(self) -> None:
        """Clean up all clients' resources."""
        await self.completions.close()
