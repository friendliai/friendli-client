# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow File Service."""

from __future__ import annotations

from string import Template
from typing import Any, Dict
from uuid import UUID

from periflow.client.base import (
    Client,
    GroupRequestMixin,
    ProjectRequestMixin,
    UserRequestMixin,
    safe_request,
)


class FileClient(Client[UUID]):
    """File client service."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_mr_uri("files/"))

    def get_misc_file_upload_url(self, misc_file_id: UUID) -> str:
        """Get an URL to upload file.

        Args:
            misc_file_id (UUID): Misc file ID to upload.

        Returns:
            str: An uploadable URL.

        """
        response = safe_request(self.post, err_prefix="Failed to get file upload URL.")(
            path=f"{misc_file_id}/upload/"
        )
        return response.json()["upload_url"]

    def get_misc_file_download_url(self, misc_file_id: UUID) -> str:
        """Get an URL to download file.

        Args:
            misc_file_id (UUID): Misc file ID to download.

        Returns:
            Dict[str, Any]: A downloadable URL.

        """
        response = safe_request(
            self.post, err_prefix="Failed to get file download URL."
        )(path=f"{misc_file_id}/download/")
        return response.json()["download_url"]

    def make_misc_file_uploaded(self, misc_file_id: UUID) -> Dict[str, Any]:
        """Request to mark the file as uploaded.

        Args:
            misc_file_id (UUID): Misc file ID to change status.

        Returns:
            Dict[str, Any]: The updated file info.

        """
        response = safe_request(
            self.partial_update, err_prefix="Failed to patch the file status."
        )(pk=misc_file_id, path="uploaded/")
        return response.json()


class GroupProjectFileClient(
    Client, UserRequestMixin, GroupRequestMixin, ProjectRequestMixin
):
    """Group-shared file client."""

    def __init__(self, **kwargs):
        """Initialize organization project file client."""
        self.initialize_user()
        self.initialize_group()
        self.initialize_project()
        super().__init__(group_id=self.group_id, project_id=self.project_id, **kwargs)

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_mr_uri("orgs/$group_id/prjs/$project_id/files/")
        )

    def create_misc_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Request to create a misc file.

        Args:
            file_info (Dict[str, Any]): File info.

        Returns:
            Dict[str, Any]: Response body with the created file info.

        """
        request_data = {
            "user_id": str(self.user_id),
            **file_info,
        }
        response = safe_request(self.post, err_prefix="Failed to create file.")(
            json=request_data
        )
        return response.json()
