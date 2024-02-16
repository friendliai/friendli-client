# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Deployment SDK."""

# pylint: disable=arguments-differ

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from friendli.client.graphql.deployment import DeploymentGqlClient
from friendli.schema.resource.v1.deployment import Deployment
from friendli.sdk.resource.base import ResourceAPI


@dataclass
class DeploymentAPI(ResourceAPI[DeploymentGqlClient, Deployment, str]):
    """Deployment resource API."""

    client: DeploymentGqlClient

    @property
    def _resource_model(self) -> Type[Deployment]:
        return Deployment

    def create(
        self,
        *,
        name: str,
        gpu_type: str,
        num_gpus: int,
        backbone_eid: str,
        adapter_eids: Optional[List[str]] = None,
        launch_config: Optional[Dict[str, Any]] = None,
    ) -> Deployment:
        """Creates a new deployment."""
        data = self.client.create(
            project_eid=self._get_project_id(),
            name=name,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            backbone_eid=backbone_eid,
            adapter_eids=adapter_eids or [],
            launch_config=launch_config or {},
        )
        deployment = self._model_parse(data)

        return deployment

    def get(self, eid: str, *args, **kwargs) -> Deployment:
        """Get information of a deployment."""
        raise NotImplementedError

    def list(self) -> List[Deployment]:
        """List deployments."""
        data = self.client.get_deployments(project_eid=self._get_project_id())
        deployments = self._model_parse(data)
        return deployments
