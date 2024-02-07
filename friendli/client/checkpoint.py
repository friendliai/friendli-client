# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-many-arguments

"""Friendli Checkpoint Clients."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from friendli.client.base import (
    Client,
    GroupRequestMixin,
    ProjectRequestMixin,
    UploadableClient,
    UserRequestMixin,
)
from friendli.enums import CheckpointCategory, StorageType


class CheckpointClient(Client[UUID]):
    """Checkpoint client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_mr_uri("models/")

    def get_checkpoint(self, checkpoint_id: UUID) -> Dict[str, Any]:
        """Get a checkpoint info."""
        data = self.retrieve(pk=checkpoint_id)
        return data

    def get_first_checkpoint_form(self, checkpoint_id: UUID) -> UUID:
        """Get the first form of the checkpoint."""
        data = self.retrieve(pk=checkpoint_id)
        return UUID(data["forms"][0]["id"])

    def activate_checkpoint(self, checkpoint_id: UUID) -> Dict[str, Any]:
        """Make checkpoint status active."""
        data = self.partial_update(pk=checkpoint_id, json={"status": "Active"})
        return data

    def delete_checkpoint(self, checkpoint_id: UUID) -> None:
        """Delete a checkpoint."""
        self.delete(pk=checkpoint_id)

    def restore_checkpoint(self, checkpoint_id: UUID) -> Dict[str, Any]:
        """Restore a soft-deleted checkpoint."""
        data = self.post(path=f"{checkpoint_id}/restore/")
        return data


class CheckpointFormClient(UploadableClient[UUID]):
    """Checkpoint form client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_mr_uri("model_forms/")

    def update_checkpoint_files(
        self,
        ckpt_form_id: UUID,
        files: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update checkpoint file metadata."""
        data = self.partial_update(pk=ckpt_form_id, json={"files": files})
        return data

    def get_checkpoint_download_urls(self, ckpt_form_id: UUID) -> List[Dict[str, Any]]:
        """Get presigned URLs to download a model checkpoint."""
        data = self.retrieve(pk=ckpt_form_id, path="download/")
        return data["files"]


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
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_mr_uri("orgs/$group_id/prjs/$project_id/models/")

    def list_checkpoints(
        self, category: Optional[CheckpointCategory], limit: int, deleted: bool
    ) -> List[Dict[str, Any]]:
        """List checkpoints."""
        params = {}
        if category is not None:
            params["category"] = category.value
        if deleted:
            params["status"] = "deleted"

        checkpoints = self.list(
            pagination=True,
            limit=limit,
            params=params,
        )
        return checkpoints

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
        data = self.post(json=request_data)
        return data
