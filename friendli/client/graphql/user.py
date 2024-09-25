# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli User GQL Clients."""

from __future__ import annotations

from typing import Any, Dict, List

from friendli.client.graphql.base import GqlClient

GetCurrentUserInfoOp = """
query GetCurrentUserInfo {
  clientSession {
    user {
      id
      name
      email
    }
  }
}
"""

ListTeamsOp = """
query ListTeams {
  clientSession {
    user {
      teams {
        edges {
          node {
            id
            name
            dedicatedSubplan {
              tier
            }
          }
        }
      }
    }
  }
}
"""

GetUserDefaultTeamOp = """
query GetUserDefaultTeam {
  clientUser {
    teamDefault {
      team {
        id
        name
      }
    }
  }
}
"""


class UserGqlClient(GqlClient):
    """User gql client."""

    def get_current_user_info(self) -> Dict[str, Any]:
        """Get current user info."""
        response = self.run(query=GetCurrentUserInfoOp)
        return response["clientSession"]["user"]

    def get_teams(self) -> List[Dict[str, Any]]:
        """List user teams."""
        response = self.run(query=ListTeamsOp)
        return response["clientSession"]["user"]["teams"]["edges"]

    def get_team_ids(self) -> List[str]:
        """List team IDs."""
        response = self.run(query=ListTeamsOp)
        return [
            edge["node"]["id"]
            for edge in response["clientSession"]["user"]["teams"]["edges"]
        ]

    def get_default_team(self) -> Dict[str, Any]:
        """Get a default team."""
        response = self.run(query=GetUserDefaultTeamOp)
        return response["clientUser"]["teamDefault"]["team"]
