# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli GQL Client Service."""

from __future__ import annotations

from typing import Any, Dict, Optional

from friendli.client.base import Client


class GqlClient(Client):
    """Base interface of graphql client to Friendli system."""

    @property
    def url_path(self) -> str:
        """URL path template to render."""
        return self.url_provider.get_web_backend_uri("api/graphql")

    def run(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run graphql."""
        return self.post(
            json={
                "query": query,
                "variables": variables,
            }
        )["data"]
