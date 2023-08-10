# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Catalog SDK."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from periflow.client.catalog import CatalogClient
from periflow.schema.resource.v1.checkpoint import V1Catalog
from periflow.sdk.resource.base import ResourceAPI


class Catalog(ResourceAPI[V1Catalog, UUID]):
    """Catalog resource API."""

    @staticmethod
    def list(name: Optional[str] = None, limit: int = 20) -> List[V1Catalog]:
        """Lists public checkpoints in the catalog.

        Args:
            name (Optional[str], optional): The name of public checkpoint as a search key. Defaults to None.
            limit (int, optional): The maximum number of retrieved results. Defaults to 20.

        Returns:
            List[V1Catalog]: A list of retrieved public checkpoints from the catalog.

        """
        client = CatalogClient()
        catalogs = [
            V1Catalog.model_validate(raw_catalog)
            for raw_catalog in client.list_catalogs(name=name, limit=limit)
        ]
        return catalogs
