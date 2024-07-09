# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Sync User resource."""

from __future__ import annotations

from ....schema import WhoamiResponse
from ...translator import translate_exception
from ._base import ResourceBase


class UserResource(ResourceBase):
    """User resource for Friendli Suite API."""

    @translate_exception
    def whoami(self) -> WhoamiResponse:
        """Check current user information."""
        resp = self._sdk.http_client.get("/api/auth/whoami")
        resp.raise_for_status()
        return WhoamiResponse.model_validate_json(resp.text)
