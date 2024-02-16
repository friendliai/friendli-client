# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli User GQL Clients."""

from __future__ import annotations

from typing import Any, Dict, List

from friendli.client.graphql.base import GqlClient

CurrUserInfoGql = """
query GetClientSession {
  clientSession {
    user {
      id
      name
      email
    }
  }
}
"""

UserTeamsGql = """
query GetClientTeams {
  clientSession {
    user {
      teams {
        edges {
          node {
            id
            name
            state
          }
        }
      }
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

    def get_teams(self) -> List[Dict[str, Any]]:
        """List user teams."""
        response = self.run(query=UserTeamsGql)
        return response["clientSession"]["user"]["teams"]["edges"]
