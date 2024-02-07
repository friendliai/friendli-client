# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli User Clients."""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import UUID

from friendli.client.base import Client, GroupRequestMixin, UserRequestMixin
from friendli.enums import GroupRole, ProjectRole


class UserMFAClient(Client):
    """User MFA client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("mfa")

    def initiate_mfa(self, mfa_type: str, mfa_token: str) -> None:
        """Authenticate by MFA token."""
        self.bare_post(path=f"challenge/{mfa_type}", headers={"x-mfa-token": mfa_token})


class UserSignUpClient(Client):
    """User sign-up client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("pf_user/self_signup")

    def verify(self, token: str, key: str) -> None:
        """Verify the email account with the token to sign up."""
        self.bare_post(path="confirm", json={"email_token": token, "key": key})


class UserClient(Client, UserRequestMixin):
    """User client."""

    def __init__(self, **kwargs) -> None:
        """Initialize user client."""
        self.initialize_user()
        super().__init__(**kwargs)

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("pf_user")

    def change_password(self, old_password: str, new_password: str) -> None:
        """Change password."""
        self.update(
            pk=self.user_id,
            path="password",
            json={"old_password": old_password, "new_password": new_password},
        )

    def set_group_privilege(
        self, pf_group_id: UUID, pf_user_id: UUID, privilege_level: GroupRole
    ) -> None:
        """Update user role in the orgnaization."""
        self.partial_update(
            pk=pf_user_id,
            path=f"pf_group/{pf_group_id}/privilege_level",
            json={"privilege_level": privilege_level.value},
        )

    def get_project_membership(self, pf_project_id: UUID) -> Dict[str, Any]:
        """Get the project membership info of the user."""
        data = self.retrieve(
            pk=self.user_id,
            path=f"pf_project/{pf_project_id}",
        )
        return data

    def add_to_project(
        self, pf_user_id: UUID, pf_project_id: UUID, access_level: ProjectRole
    ) -> None:
        """Add a new member to a project."""
        self.post(
            path=f"{pf_user_id}/pf_project/{pf_project_id}",
            json={"access_level": access_level.value},
        )

    def delete_from_org(self, pf_user_id: UUID, pf_org_id: UUID) -> None:
        """Delete a member from the organization."""
        self.delete(pk=pf_user_id, path=f"pf_group/{pf_org_id}")

    def delete_from_project(self, pf_user_id: UUID, pf_project_id: UUID) -> None:
        """Delete a member from the organization."""
        self.delete(pk=pf_user_id, path=f"pf_project/{pf_project_id}")

    def set_project_privilege(
        self, pf_user_id: UUID, pf_project_id: UUID, access_level: ProjectRole
    ) -> None:
        """Set a project-level role to a user."""
        self.partial_update(
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
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("pf_user/$pf_user_id/pf_group")

    def get_group_info(self) -> Dict[str, Any]:
        """Get organization info where user belongs to."""
        data = self.list(pagination=False)
        return data[0]


class UserGroupProjectClient(Client, UserRequestMixin, GroupRequestMixin):
    """User organization project client."""

    def __init__(self, **kwargs):
        """Initialize user organization project client."""
        self.initialize_user()
        self.initialize_group()
        super().__init__(pf_user_id=self.user_id, pf_group_id=self.group_id, **kwargs)

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri(
            "pf_user/$pf_user_id/pf_group/$pf_group_id/pf_project"
        )

    def list_projects(self) -> List[Dict[str, Any]]:
        """List projects in the organization."""
        data = self.list(pagination=True)
        return data


class UserAccessKeyClient(Client, UserRequestMixin):
    """User access key client."""

    def __init__(self, **kwargs):
        """Initialize user access key client."""
        self.initialize_user()
        super().__init__(pf_user_id=self.user_id, **kwargs)

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("pf_user")

    def create_access_key(self, name: str) -> Dict[str, Any]:
        """Create a new access key."""
        data = self.post(path=f"{self.user_id}/api_key", json={"name": name})
        return data

    def delete_access_key(self, access_key_id: str) -> None:
        """Revoke an access key."""
        self.delete(pk=None, path=f"api_key/{access_key_id}")

    def list_access_keys(self) -> List[Dict[str, Any]]:
        """List access keys."""
        data = self.list(path=f"{self.user_id}/api_key", pagination=False)
        return data
