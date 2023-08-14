# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Catalog Client."""

from __future__ import annotations

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
from periflow.enums import CatalogImportMethod
from periflow.utils.request import paginated_get


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
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_mr_uri("catalogs/"))

    def get_catalog(self, catalog_id: UUID) -> Dict[str, Any]:
        """Get a public checkpoint in catalog."""
        response = safe_request(
            self.retrieve,
            err_prefix="Failed to get info of public checkpoint in catalog",
        )(pk=catalog_id)
        return response.json()

    def list_catalogs(self, name: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """List public checkpoints in catalog."""
        request_data = {}
        if name is not None:
            request_data["name"] = name

        resp_getter = safe_request(
            self.list, err_prefix="Failed to get public checkpoints in catalog."
        )
        return paginated_get(
            resp_getter,
            **request_data,
            limit=limit,
            sort_by_popularity="descending",
        )

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
        response = safe_request(
            self.post,
            err_prefix="Failed to import public checkpoint from catalog to project",
        )(path=f"{catalog_id}/try_out/", json=request_data)
        return response.json()
