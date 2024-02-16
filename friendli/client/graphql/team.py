# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Team GQL Clients."""

from __future__ import annotations

from typing import Any, Dict, List

from friendli.client.graphql.base import GqlClient

TeamProjectsGql = """
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
        response = self.run(query=TeamProjectsGql, variables={"input": team_id})
        return response["clientTeam"]["projects"]["edges"]
