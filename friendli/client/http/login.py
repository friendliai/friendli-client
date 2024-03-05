# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Login Client."""

from __future__ import annotations

from typing import Tuple

from friendli.client.http.base import HttpClient


class LoginClient(HttpClient):
    """Login client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_web_backend_uri("/api/auth/login")

    def login(self, email: str, pwd: str) -> Tuple[str, str]:
        """Send request to sign in with email and password."""
        payload = {
            "email": email,
            "password": pwd,
        }
        headers = {
            "Accept": "application/json",
            "rid": "anti-csrf",
            "st-auth-mode": "header",
        }
        resp = self.bare_post(json=payload, headers=headers)
        access_token = resp.headers["st-access-token"]
        refresh_token = resp.headers["st-refresh-token"]
        return access_token, refresh_token
