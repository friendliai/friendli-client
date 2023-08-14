# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-many-arguments

"""PeriFlow GroupClient Service."""

from __future__ import annotations

import json
import uuid
from string import Template
from typing import Any, Dict, List, Optional
from uuid import UUID

from periflow.client.base import (
    Client,
    GroupRequestMixin,
    ProjectRequestMixin,
    UserRequestMixin,
    safe_request,
)
from periflow.enums import CheckpointCategory, StorageType
from periflow.utils.request import paginated_get


class GroupClient(Client):
    """Organization client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_auth_uri("pf_group"))

    def create_group(self, name: str) -> Dict[str, Any]:
        """Create a new organization."""
        response = safe_request(
            self.post, err_prefix="Failed to post an organization."
        )(data=json.dumps({"name": name, "hosting_type": "hosted"}))
        return response.json()

    def get_group(self, pf_group_id: uuid.UUID) -> Dict[str, Any]:
        """Get the organization info."""
        response = safe_request(
            self.retrieve, err_prefix="Failed to get an organization."
        )(pk=pf_group_id)
        return response.json()

    def invite_to_group(self, pf_group_id: uuid.UUID, email: str) -> None:
        """Invite a new member to the organization by sending an email."""
        safe_request(self.post, err_prefix="Failed to send invitation")(
            path=f"{pf_group_id}/invite/signup",
            json={
                "email": email,
                "callback_path": "/api/auth/invite/callback",
            },
        )

    def list_users(self, pf_group_id: uuid.UUID) -> List[Dict[str, Any]]:
        """List all organization member info."""
        get_response_dict = safe_request(
            self.list, err_prefix="Failed to list users in organization"
        )
        return paginated_get(get_response_dict, path=f"{pf_group_id}/pf_user")


class GroupProjectClient(Client, GroupRequestMixin):
    """Organization project client."""

    def __init__(self, **kwargs):
        """Initialize group project client."""
        self.initialize_group()
        super().__init__(pf_group_id=self.group_id, **kwargs)

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_auth_uri("pf_group/$pf_group_id/pf_project")
        )

    def create_project(self, name: str) -> Dict[str, Any]:
        """Create a new project in the organization."""
        response = safe_request(self.post, err_prefix="Failed to post a project.")(
            data=json.dumps({"name": name})
        )
        return response.json()

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects in the organization."""
        get_response_dict = safe_request(
            self.list, err_prefix="Failed to list projects."
        )
        return paginated_get(get_response_dict)


class GroupProjectCheckpointClient(
    Client, UserRequestMixin, GroupRequestMixin, ProjectRequestMixin
):
    """Organization project checkpoint client."""

    def __init__(self, **kwargs):
        """Initialize organization project checkpoint client."""
        self.initialize_user()
        self.initialize_group()
        self.initialize_project()
        super().__init__(group_id=self.group_id, project_id=self.project_id, **kwargs)

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_mr_uri("orgs/$group_id/prjs/$project_id/models/")
        )

    def list_checkpoints(
        self, category: Optional[CheckpointCategory], limit: int, deleted: bool
    ) -> List[Dict[str, Any]]:
        """List checkpoints."""
        request_data = {}
        if category is not None:
            request_data["category"] = category.value
        if deleted:
            request_data["status"] = "deleted"

        get_response_dict = safe_request(
            self.list, err_prefix="Failed to list checkpoints."
        )
        return paginated_get(get_response_dict, **request_data, limit=limit)

    def create_checkpoint(
        self,
        name: str,
        vendor: StorageType,
        region: str,
        credential_id: Optional[UUID],
        iteration: Optional[int],
        storage_name: str,
        files: List[Dict[str, Any]],
        dist_config: Dict[str, Any],
        attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a new checkpoint."""
        request_data = {
            "job_id": None,
            "name": name,
            "attributes": attributes,
            "user_id": str(self.user_id),
            "secret_type": "credential",
            "secret_id": str(credential_id) if credential_id else None,
            "model_category": "USER",
            "form_category": "ORCA",
            "dist_json": dist_config,
            "vendor": vendor,
            "region": region,
            "storage_name": storage_name,
            "iteration": iteration,
            "files": files,
        }

        response = safe_request(self.post, err_prefix="Failed to post checkpoint.")(
            json=request_data
        )
        return response.json()
