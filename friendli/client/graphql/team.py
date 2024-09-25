# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Team GQL Clients."""

from __future__ import annotations

from typing import Any, Dict, List

from friendli.client.graphql.base import GqlClient

ListProjectsInTeamOp = """
query ListProjectsInTeam($input: ID!) {
  clientTeam(id: $input) {
    dedicatedSubplan {
      projects {
        edges {
          node {
            id
            name
          }
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
        response = self.run(query=ListProjectsInTeamOp, variables={"input": team_id})
        return response["clientTeam"]["dedicatedSubplan"]["projects"]["edges"]

    def get_project_ids(self, team_id: str) -> List[str]:
        """List project IDs."""
        response = self.run(query=ListProjectsInTeamOp, variables={"input": team_id})
        return [
            edge["node"]["id"]
            for edge in response["clientTeam"]["dedicatedSubplan"]["projects"]["edges"]
        ]
