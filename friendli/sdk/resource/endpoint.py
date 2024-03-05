# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Endpoint SDK."""

# pylint: disable=arguments-differ

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Type

from friendli.client.graphql.endpoint import EndpointGqlClient
from friendli.schema.resource.v1.endpoint import Endpoint
from friendli.sdk.resource.base import ResourceApi


@dataclass
class EndpointApi(ResourceApi[EndpointGqlClient, Endpoint, str]):
    """Endpoint resource API."""

    client: EndpointGqlClient

    @property
    def _resource_model(self) -> Type[Endpoint]:
        return Endpoint

    def create(
        self,
        *,
        name: str,
        model_repo: str,
        gpu_type: str,
        num_gpus: int,
    ) -> Endpoint:
        """Creates a new dedicated endpoint."""
        data = self.client.create(
            team_id=self._get_team_id(),
            project_id=self._get_project_id(),
            name=name,
            model_repo=model_repo,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
        )
        endpoint = self._model_parse(data)

        return endpoint

    def get(self, eid: str, *args, **kwargs) -> Endpoint:
        """Get information of a endpoint."""
        data = self.client.get(endpoint_id=eid)
        endpoint = self._model_parse(data)
        return endpoint

    def list(self) -> List[Endpoint]:
        """List endpoints."""
        data = self.client.list(project_id=self._get_project_id())
        endpoints = self._model_parse(data)
        return endpoints

    def terminate(self, endpoint_id: str) -> None:
        """Terminate a endpoint."""
        self.client.terminate(
            team_id=self._get_team_id(),
            project_id=self._get_project_id(),
            endpoint_id=endpoint_id,
        )
