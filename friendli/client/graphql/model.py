# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Model GQL Client."""

from __future__ import annotations

from typing import Any, Dict, List

from friendli.client.graphql.base import GqlClient

ListModelsInProjectOp = """
query ListModelsInProject($projectId: ID!) {
  dedicatedProject(id: $projectId) {
    id
    models {
      edges {
        node {
          id
          name
        }
      }
    }
  }
}
"""


class ModelGqlClient(GqlClient):
    """Model GQL client."""

    def list(self, project_id: str) -> List[Dict[str, Any]]:
        """List models in the project."""
        response = self.run(
            query=ListModelsInProjectOp, variables={"projectId": project_id}
        )
        return response["dedicatedProject"]["models"]["edges"]
