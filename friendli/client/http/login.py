# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Login Client."""

from __future__ import annotations

from typing import Tuple

from friendli.client.http.base import HttpClient
from friendli.settings import Settings


class LoginClient(HttpClient):
    """Login client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_web_backend_uri("/api/auth/login")

    def login(self, email: str, pwd: str) -> Tuple[str, str]:
        """Send request to sign in with email and password."""
        settings = self.injector.get(Settings)
        payload = {
            "email": email,
            "password": pwd,
        }
        headers = {"Accept": "application/json"}
        resp = self.bare_post(json=payload, headers=headers)
        cookies = resp.cookies
        access_token = cookies[settings.access_token_cookie_key]
        refresh_token = cookies[settings.refresh_token_cookie_key]
        return access_token, refresh_token
