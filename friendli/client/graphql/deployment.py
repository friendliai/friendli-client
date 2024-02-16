# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Deployment GQL Clients."""

from __future__ import annotations

from typing import Any, Dict, List

from friendli.client.graphql.base import GqlClient

CreateDeploymentGql = """
mutation CreateDeployment($input: InferenceDeploymentCreateInput!) {
  inferenceDeploymentCreate(input: $input) {
    id
    name
    artifactId
    gpuType
    numGpu
    status
    createdAt
    updatedAt
  }
}
"""

ProjectDeploymentsGql = """
query ClientProject($input: ID!) {
  clientProject(id: $input) {
    deployments {
      edges {
        node {
          id
          name
          artifactId
          gpuType
          numGpu
          status
          createdAt
          updatedAt
        }
      }
    }
  }
}
"""


class DeploymentGqlClient(GqlClient):
    """Deployment GQL client."""

    def create(  # pylint: disable=too-many-arguments
        self,
        project_eid: str,
        name: str,
        backbone_eid: str,
        adapter_eids: List[str],
        gpu_type: str,
        num_gpus: int,
        launch_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get current user info."""
        response = self.run(
            query=CreateDeploymentGql,
            variables={
                "input": {
                    "projectEid": project_eid,
                    "name": name,
                    "backboneEid": backbone_eid,
                    "adapterEids": adapter_eids,
                    "gpuType": gpu_type,
                    "numGpus": num_gpus,
                    "launchConfig": launch_config,
                }
            },
        )
        return response["inferenceDeploymentCreate"]

    def get_deployments(self, project_eid: str) -> List[Dict[str, Any]]:
        """List team projects."""
        response = self.run(
            query=ProjectDeploymentsGql, variables={"input": project_eid}
        )
        return response["clientProject"]["deployments"]["edges"]
