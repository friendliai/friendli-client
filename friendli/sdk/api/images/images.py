# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Images API."""

from __future__ import annotations

from typing import Optional

import httpx

from friendli.sdk.api.images.text_to_image import AsyncTextToImage, TextToImage


class Images:
    """Images API."""

    text_to_image: TextToImage

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize Images."""
        self.text_to_image = TextToImage(
            base_url=base_url, endpoint_id=endpoint_id, http_client=http_client
        )

    def close(self) -> None:
        """Clean up all clients' resources."""
        self.text_to_image.close()


class AsyncImages:
    """Asynchronous images API."""

    text_to_image: AsyncTextToImage

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize Images."""
        self.text_to_image = AsyncTextToImage(
            base_url=base_url, endpoint_id=endpoint_id, http_client=http_client
        )

    async def close(self) -> None:
        """Clean up all clients' resources."""
        await self.text_to_image.close()
