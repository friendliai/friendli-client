# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Interface for resource management SDK."""

# pylint: disable=redefined-builtin

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Type, TypeVar, Union, overload

import pydantic
from injector import inject

from friendli.client.graphql.base import GqlClient
from friendli.context import get_current_project_id, get_current_team_id
from friendli.errors import AuthorizationError
from friendli.utils.compat import model_parse

_Resource = TypeVar("_Resource", bound=pydantic.BaseModel)
_ResourceId = TypeVar("_ResourceId")
_Client = TypeVar("_Client", bound=GqlClient)


@inject
@dataclass
class ResourceAPI(ABC, Generic[_Client, _Resource, _ResourceId]):
    """Abstract class for resource APIs."""

    client: _Client

    @property
    @abstractmethod
    def _resource_model(self) -> Type[_Resource]:
        """Model type of resource."""

    @abstractmethod
    def create(self, *args, **kwargs) -> _Resource:
        """Creates a resource."""

    @abstractmethod
    def get(self, eid: _ResourceId, *args, **kwargs) -> _Resource:
        """Gets a specific resource."""

    @abstractmethod
    def list(self, *args, **kwargs) -> List[_Resource]:
        """Lists reousrces."""

    @overload
    def _model_parse(self, data: Dict[str, Any]) -> _Resource:
        ...

    @overload
    def _model_parse(self, data: List[Dict[str, Any]]) -> List[_Resource]:
        ...

    def _model_parse(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[_Resource, List[_Resource]]:
        if isinstance(data, list):
            return [model_parse(self._resource_model, entry["node"]) for entry in data]
        return model_parse(self._resource_model, data)

    def _get_project_id(self) -> str:
        project_id = get_current_project_id()
        if project_id is None:
            raise AuthorizationError(
                "Project to run as is not configured. Set 'FRIENDLI_PROJECT' "
                "environment variable, or use a CLI command 'friendli project switch'."
            )
        return project_id

    def _get_team_id(self) -> str:
        team_id = get_current_team_id()
        if team_id is None:
            raise AuthorizationError(
                "Team to run as is not configured. Set 'FRIENDLI_TEAM' environment "
                "variable, or use a CLI command 'friendli team switch'."
            )
        return team_id
