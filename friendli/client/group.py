# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-many-arguments

"""Friendli Group Clients."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from friendli.client.base import Client, GroupRequestMixin


class GroupClient(Client):
    """Organization client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("pf_group")

    def create_group(self, name: str) -> Dict[str, Any]:
        """Create a new organization."""
        data = self.post(
            data=json.dumps({"name": name, "hosting_type": "hosted"}),
        )
        return data

    def get_group(self, pf_group_id: uuid.UUID) -> Dict[str, Any]:
        """Get the organization info."""
        data = self.retrieve(pk=pf_group_id)
        return data

    def invite_to_group(self, pf_group_id: uuid.UUID, email: str) -> None:
        """Invite a new member to the organization by sending an email."""
        self.post(
            path=f"{pf_group_id}/invite/signup",
            json={
                "email": email,
                "callback_path": "/api/auth/invite/callback",
            },
        )

    def list_users(self, pf_group_id: uuid.UUID) -> List[Dict[str, Any]]:
        """List all organization member info."""
        users = self.list(
            path=f"{pf_group_id}/pf_user",
            pagination=True,
        )
        return users


class GroupProjectClient(Client, GroupRequestMixin):
    """Organization project client."""

    def __init__(self, **kwargs):
        """Initialize group project client."""
        self.initialize_group()
        super().__init__(pf_group_id=self.group_id, **kwargs)

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("pf_group/$pf_group_id/pf_project")

    def create_project(self, name: str) -> Dict[str, Any]:
        """Create a new project in the organization."""
        data = self.post(
            data=json.dumps({"name": name}),
        )
        return data

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects in the organization."""
        projects = self.list(
            pagination=True,
        )
        return projects
