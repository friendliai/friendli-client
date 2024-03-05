# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Team GQL Clients."""

from __future__ import annotations

from typing import Any, Dict, List

from friendli.client.graphql.base import GqlClient

TeamProjectsOp = """
query ClientTeam($input: ID!) {
  clientTeam(id: $input) {
    projects {
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
"""


class TeamGqlClient(GqlClient):
    """Team gql client."""

    def get_projects(self, team_id: str) -> List[Dict[str, Any]]:
        """List team projects."""
        response = self.run(query=TeamProjectsOp, variables={"input": team_id})
        return response["clientTeam"]["projects"]["edges"]

    def get_project_ids(self, team_id: str) -> List[str]:
        """List project IDs."""
        response = self.run(query=TeamProjectsOp, variables={"input": team_id})
        return [
            edge["node"]["id"] for edge in response["clientTeam"]["projects"]["edges"]
        ]
