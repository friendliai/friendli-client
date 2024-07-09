# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Sync Auth strategies."""

from __future__ import annotations

from typing import Generator

from httpx import Auth, Request, Response


class BearerAuth(Auth):
    """Authenticates using Bearer token."""

    def __init__(self, token: str) -> None:
        """Initialize."""
        self._auth_header = f"Bearer {token}"

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        """Sync auth flow."""
        request.headers["Authorization"] = self._auth_header
        yield request
