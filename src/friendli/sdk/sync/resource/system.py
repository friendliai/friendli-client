# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Sync System resource."""

from __future__ import annotations

from ....schema import VersionResponse
from ...translator import translate_exception
from ._base import ResourceBase


class SystemResource(ResourceBase):
    """System resource for Friendli Suite API."""

    @translate_exception
    def ping(self) -> None:
        """Ping the Friendli Suite API."""
        # For ping request, set the timeout to 1 second. This is to ensure the API
        # server is up and responding within a reasonable timeframe.
        resp = self._sdk.http_client.get("/api/status/", timeout=1)
        resp.raise_for_status()

    @translate_exception
    def version(self) -> VersionResponse:
        """Get the version of the Friendli Suite API."""
        resp = self._sdk.http_client.get("/api/version/")
        resp.raise_for_status()
        return VersionResponse.model_validate_json(resp.text)
