# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow ProjectClient Service."""

# pylint: disable=too-many-arguments

from __future__ import annotations

from string import Template
from typing import Any, Dict, List, Optional
from uuid import UUID

from requests import HTTPError

from periflow.client.base import Client, ProjectRequestMixin, safe_request
from periflow.enums import CredType
from periflow.utils.format import secho_error_and_exit
from periflow.utils.maps import cred_type_map
from periflow.utils.request import paginated_get


def find_project_id(projects: List[Dict[str, Any]], project_name: str) -> UUID:
    """Find an ID of project among a list of projects."""
    for project in projects:
        if project["name"] == project_name:
            return UUID(project["id"])
    secho_error_and_exit(f"No project exists with name {project_name}.")


class ProjectClient(Client[UUID]):
    """Project client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_auth_uri("pf_project"))

    def get_project(self, pf_project_id: UUID) -> Dict[str, Any]:
        """Get project info."""
        response = safe_request(self.retrieve, err_prefix="Failed to get a project.")(
            pk=pf_project_id
        )
        return response.json()

    def check_project_membership(self, pf_project_id: UUID) -> bool:
        """Check accessibility to the project."""
        try:
            self.retrieve(pf_project_id)
        except HTTPError:
            return False
        return True

    def delete_project(self, pf_project_id: UUID) -> None:
        """Delete a project."""
        safe_request(self.delete, err_prefix="Failed to delete a project.")(
            pk=pf_project_id
        )

    def list_users(self, pf_project_id: UUID) -> List[Dict[str, Any]]:
        """List all project members."""
        get_response_dict = safe_request(
            self.list, err_prefix="Failed to list users in the current project"
        )
        return paginated_get(get_response_dict, path=f"{pf_project_id}/pf_user")


class ProjectCredentialClient(Client, ProjectRequestMixin):
    """Project credential client."""

    def __init__(self, **kwargs):
        """Initialize project credential client."""
        self.initialize_project()
        super().__init__(project_id=self.project_id, **kwargs)

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_auth_uri("pf_project/$project_id/credential")
        )

    def list_credentials(
        self, cred_type: Optional[CredType] = None
    ) -> List[Dict[str, Any]]:
        """List all credentials."""
        params = {}
        if cred_type is not None:
            params["type"] = cred_type_map[cred_type]
        response = safe_request(
            self.list, err_prefix=f"Failed to list credential for {cred_type}."
        )(params=params)
        return response.json()

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
        response = safe_request(
            self.post, err_prefix="Failed to create user credential."
        )(json=request_data)
        return response.json()
