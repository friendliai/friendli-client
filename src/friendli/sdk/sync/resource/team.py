# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Sync Team resource."""

from __future__ import annotations

from ...graphql.api import (
    BidirectionalConnectionInput,
    ClientUserTeamSortsInput,
    UserContextResult,
    UserContextVariables,
)
from ._base import ResourceBase


class TeamResource(ResourceBase):
    """Team resource for Friendli Suite API."""

    def list(self) -> UserContextResult:
        """List teams."""
        variables = UserContextVariables(
            conn=BidirectionalConnectionInput(first=5, skip=0),
            sorts=ClientUserTeamSortsInput(ascending=True, sortBy="joined_at"),
        )
        return self._sdk.gql_client.user_context(variables)
