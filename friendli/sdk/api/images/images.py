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
        deployment_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize Images."""
        self.text_to_image = TextToImage(
            deployment_id=deployment_id, endpoint=endpoint, client=client
        )


class AsyncImages:
    """Asynchronous images API."""

    text_to_image: AsyncTextToImage

    def __init__(
        self,
        deployment_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize Images."""
        self.text_to_image = AsyncTextToImage(
            deployment_id=deployment_id, endpoint=endpoint, client=client
        )
