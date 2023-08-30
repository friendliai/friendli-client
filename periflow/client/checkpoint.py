# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CheckpointClient Service."""

from __future__ import annotations

from string import Template
from typing import Any, Dict, List
from uuid import UUID

from periflow.client.base import Client, UploadableClient, safe_request


class CheckpointClient(Client[UUID]):
    """Checkpoint client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_mr_uri("models/"))

    def get_checkpoint(self, checkpoint_id: UUID) -> Dict[str, Any]:
        """Get a checkpoint info."""
        response = safe_request(
            self.retrieve, err_prefix="Failed to get info of checkpoint"
        )(pk=checkpoint_id)
        return response.json()

    def get_first_checkpoint_form(self, checkpoint_id: UUID) -> UUID:
        """Get the first form of the checkpoint."""
        response = safe_request(
            self.retrieve, err_prefix="Failed to get info of checkpoint."
        )(pk=checkpoint_id)
        return UUID(response.json()["forms"][0]["id"])

    def activate_checkpoint(self, checkpoint_id: UUID) -> Dict[str, Any]:
        """Make checkpoint status active."""
        response = safe_request(
            self.partial_update, err_prefix="Failed to activate checkpoint."
        )(pk=checkpoint_id, json={"status": "Active"})
        return response.json()

    def delete_checkpoint(self, checkpoint_id: UUID) -> None:
        """Delete a checkpoint."""
        safe_request(self.delete, err_prefix="Failed to delete checkpoint.")(
            pk=checkpoint_id
        )

    def restore_checkpoint(self, checkpoint_id: UUID) -> Dict[str, Any]:
        """Restore a soft-deleted checkpoint."""
        response = safe_request(self.post, err_prefix="Fail to restore checkpoint.")(
            path=f"{checkpoint_id}/restore/"
        )
        return response.json()


class CheckpointFormClient(UploadableClient[UUID]):
    """Checkpoint form client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_mr_uri("model_forms/"))

    def update_checkpoint_files(
        self,
        ckpt_form_id: UUID,
        files: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update checkpoint file metadata."""
        response = safe_request(
            self.partial_update, err_prefix="Cannot update checkpoint."
        )(pk=ckpt_form_id, json={"files": files})
        return response.json()

    def get_checkpoint_download_urls(self, ckpt_form_id: UUID) -> List[Dict[str, Any]]:
        """Get presigned URLs to download a model checkpoint."""
        response = safe_request(
            self.retrieve, err_prefix="Failed to get presigned URLs."
        )(pk=ckpt_form_id, path="download/")
        return response.json()["files"]
