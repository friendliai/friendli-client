# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Model SDK."""

# pylint: disable=arguments-differ

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Type

from friendli.client.graphql.model import ModelGqlClient
from friendli.schema.resource.v1.model import Model
from friendli.sdk.resource.base import ResourceApi


@dataclass
class ModelApi(ResourceApi[ModelGqlClient, Model, str]):
    """Model resource API."""

    client: ModelGqlClient

    @property
    def _resource_model(self) -> Type[Model]:
        return Model

    def create(self, *args, **kwargs) -> Model:
        """Create model."""
        raise NotImplementedError

    def get(self, eid: str, *args, **kwargs) -> Model:
        """Get model information."""
        raise NotImplementedError

    def list(self) -> List[Model]:
        """List models."""
        data = self.client.list(project_id=self._get_project_id())
        models = self._model_parse(data)
        return models
