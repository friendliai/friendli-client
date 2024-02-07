# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli User GQL Clients."""

from __future__ import annotations

from typing import Any, Dict

from friendli.graphql.base import GqlClient

CurrUserInfoGql = """
query GetclientSession {
  clientSession {
    user {
      id
      name
      email
    }
  }
}
"""


class UserGqlClient(GqlClient):
    """User gql client."""

    def get_current_user_info(self) -> Dict[str, Any]:
        """Get current user info."""
        response = self.run(query=CurrUserInfoGql)
        return response["clientSession"]["user"]
