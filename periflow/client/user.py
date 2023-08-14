# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow UserClient Service."""

from __future__ import annotations

from string import Template
from typing import Any, Dict, List
from uuid import UUID

from periflow.client.base import (
    Client,
    GroupRequestMixin,
    UserRequestMixin,
    safe_request,
)
from periflow.enums import GroupRole, ProjectRole
from periflow.utils.request import paginated_get


class UserMFAClient(Client):
    """User MFA client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_auth_uri("mfa"))

    def initiate_mfa(self, mfa_type: str, mfa_token: str) -> None:
        """Authenticate by MFA token."""
        safe_request(self.bare_post, err_prefix="Failed to verify MFA token.")(
            path=f"challenge/{mfa_type}", headers={"x-mfa-token": mfa_token}
        )


class UserSignUpClient(Client):
    """User sign-up client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_auth_uri("pf_user/self_signup"))

    def verify(self, token: str, key: str) -> None:
        """Verify the email account with the token to sign up."""
        safe_request(self.bare_post, err_prefix="Failed to verify")(
            path="confirm", json={"email_token": token, "key": key}
        )


class UserClient(Client, UserRequestMixin):
    """User client."""

    def __init__(self, **kwargs) -> None:
        """Initialize user client."""
        self.initialize_user()
        super().__init__(**kwargs)

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_auth_uri("pf_user"))

    def change_password(self, old_password: str, new_password: str) -> None:
        """Change password."""
        safe_request(self.update, err_prefix="Failed to change password.")(
            pk=self.user_id,
            path="password",
            json={"old_password": old_password, "new_password": new_password},
        )

    def set_group_privilege(
        self, pf_group_id: UUID, pf_user_id: UUID, privilege_level: GroupRole
    ) -> None:
        """Update user role in the orgnaization."""
        safe_request(
            self.partial_update, err_prefix="Failed to update privilege level in group"
        )(
            pk=pf_user_id,
            path=f"pf_group/{pf_group_id}/privilege_level",
            json={"privilege_level": privilege_level.value},
        )

    def get_project_membership(self, pf_project_id: UUID) -> Dict[str, Any]:
        """Get the project membership info of the user."""
        response = safe_request(
            self.retrieve, err_prefix="Failed identify member in project"
        )(
            pk=self.user_id,
            path=f"pf_project/{pf_project_id}",
        )
        return response.json()

    def add_to_project(
        self, pf_user_id: UUID, pf_project_id: UUID, access_level: ProjectRole
    ) -> None:
        """Add a new member to a project."""
        safe_request(self.post, err_prefix="Failed to add user to project")(
            path=f"{pf_user_id}/pf_project/{pf_project_id}",
            json={"access_level": access_level.value},
        )

    def delete_from_org(self, pf_user_id: UUID, pf_org_id: UUID) -> None:
        """Delete a member from the organization."""
        safe_request(self.delete, err_prefix="Failed to remove user from organization")(
            pk=pf_user_id, path=f"pf_group/{pf_org_id}"
        )

    def delete_from_project(self, pf_user_id: UUID, pf_project_id: UUID) -> None:
        """Delete a member from the organization."""
        safe_request(self.delete, err_prefix="Failed to remove user from proejct")(
            pk=pf_user_id, path=f"pf_project/{pf_project_id}"
        )

    def set_project_privilege(
        self, pf_user_id: UUID, pf_project_id: UUID, access_level: ProjectRole
    ) -> None:
        """Set a project-level role to a user."""
        safe_request(
            self.partial_update,
            err_prefix="Failed to update privilege level in project",
        )(
            pk=pf_user_id,
            path=f"pf_project/{pf_project_id}/access_level",
            json={"access_level": access_level.value},
        )


class UserGroupClient(Client, UserRequestMixin):
    """User organization client."""

    def __init__(self, **kwargs):
        """Initialize user organization client."""
        self.initialize_user()
        super().__init__(pf_user_id=self.user_id, **kwargs)

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_auth_uri("pf_user/$pf_user_id/pf_group"))

    def get_group_info(self) -> Dict[str, Any]:
        """Get organization info where user belongs to."""
        response = safe_request(self.list, err_prefix="Failed to get my group info.")()
        return response.json()[0]


class UserGroupProjectClient(Client, UserRequestMixin, GroupRequestMixin):
    """User organization project client."""

    def __init__(self, **kwargs):
        """Initialize user organization project client."""
        self.initialize_user()
        self.initialize_group()
        super().__init__(pf_user_id=self.user_id, pf_group_id=self.group_id, **kwargs)

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_auth_uri(
                "pf_user/$pf_user_id/pf_group/$pf_group_id/pf_project"
            )
        )

    def list_projects(self) -> List[Dict[str, Any]]:
        """List projects in the organization."""
        get_response_dict = safe_request(
            self.list, err_prefix="Failed to list projects."
        )
        return paginated_get(get_response_dict)


class UserAccessKeyClient(Client, UserRequestMixin):
    """User access key client."""

    def __init__(self, **kwargs):
        """Initialize user access key client."""
        self.initialize_user()
        super().__init__(pf_user_id=self.user_id, **kwargs)

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_auth_uri("pf_user"))

    def create_access_key(self, name: str) -> Dict[str, Any]:
        """Create a new access key."""
        response = safe_request(
            self.post, err_prefix="Failed to create new access key."
        )(path=f"{self.user_id}/api_key", json={"name": name})

        return response.json()

    def delete_access_key(self, access_key_id: str) -> None:
        """Revoke an access key."""
        safe_request(self.delete, err_prefix="Failed to delete access key")(
            pk=None, path=f"api_key/{access_key_id}"
        )

    def list_access_keys(self) -> List[Dict[str, Any]]:
        """List access keys."""
        response = safe_request(
            self.list, err_prefix="Failed to list available access keys."
        )(path=f"{self.user_id}/api_key")
        return response.json()
