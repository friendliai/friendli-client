# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Project Clients."""

# pylint: disable=too-many-arguments

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from requests import HTTPError

from friendli.client.base import Client, ProjectRequestMixin
from friendli.enums import CredType
from friendli.errors import NotFoundError
from friendli.utils.maps import cred_type_map


def find_project_id(projects: List[Dict[str, Any]], project_name: str) -> UUID:
    """Find an ID of project among a list of projects."""
    for project in projects:
        if project["name"] == project_name:
            return UUID(project["id"])
    raise NotFoundError(f"No project exists with name {project_name}.")


class ProjectClient(Client[UUID]):
    """Project client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("pf_project")

    def get_project(self, pf_project_id: UUID) -> Dict[str, Any]:
        """Get project info."""
        data = self.retrieve(pk=pf_project_id)
        return data

    def check_project_membership(self, pf_project_id: UUID) -> bool:
        """Check accessibility to the project."""
        try:
            self.retrieve(pf_project_id)
        except HTTPError:
            return False
        return True

    def delete_project(self, pf_project_id: UUID) -> None:
        """Delete a project."""
        self.delete(pk=pf_project_id)

    def list_users(self, pf_project_id: UUID) -> List[Dict[str, Any]]:
        """List all project members."""
        users = self.list(
            path=f"{pf_project_id}/pf_user",
            pagination=True,
        )
        return users


class ProjectCredentialClient(Client, ProjectRequestMixin):
    """Project credential client."""

    def __init__(self, **kwargs):
        """Initialize project credential client."""
        self.initialize_project()
        super().__init__(project_id=self.project_id, **kwargs)

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("pf_project/$project_id/credential")

    def list_credentials(
        self, cred_type: Optional[CredType] = None
    ) -> List[Dict[str, Any]]:
        """List all credentials."""
        params = {}
        if cred_type is not None:
            params["type"] = cred_type_map[cred_type]
        data = self.list(pagination=False, params=params)
        return data

    def create_credential(
        self, cred_type: CredType, name: str, type_version: int, value: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new credential."""
        type_name = cred_type_map[cred_type]
        request_data = {
            "type": type_name,
            "name": name,
            "type_version": type_version,
            "value": value,
        }
        data = self.post(json=request_data)
        return data
