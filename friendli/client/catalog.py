# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Catalog Client."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from friendli.client.base import (
    Client,
    GroupRequestMixin,
    ProjectRequestMixin,
    UserRequestMixin,
)
from friendli.enums import CatalogImportMethod


class CatalogClient(
    Client[UUID], GroupRequestMixin, ProjectRequestMixin, UserRequestMixin
):
    """Catalog client."""

    def __init__(self, **kwargs):
        """Initialize catalog client."""
        self.initialize_user()
        self.initialize_group()
        self.initialize_project()
        super().__init__(**kwargs)

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_mr_uri("catalogs/")

    def get_catalog(self, catalog_id: UUID) -> Dict[str, Any]:
        """Get a public checkpoint in catalog."""
        data = self.retrieve(pk=catalog_id)
        return data

    def list_catalogs(self, name: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """List public checkpoints in catalog."""
        params = {
            "sort_by_popularity": "descending",
            "statuses": "Active",
        }
        if name is not None:
            params["name"] = name

        data = self.list(pagination=True, limit=limit, params=params)
        return data

    def try_out(
        self, catalog_id: UUID, name: str, method: CatalogImportMethod
    ) -> Dict[str, Any]:
        """Import a public checkpoint to a project.

        Args:
            catalog_id (UUID): ID of public checkpoint in catalog.
            name (str): The name of model to create in the project.
            method (CatalogImportMethod): Import method.

        Returns:
            Dict[str, Any]: Information of the created checkpoint.

        """
        request_data = {
            "name": name,
            "organization_id": str(self.group_id),
            "project_id": str(self.project_id),
            "user_id": str(self.user_id),
            "import_method": method.value,
        }
        data = self.post(path=f"{catalog_id}/try_out/", json=request_data)
        return data
