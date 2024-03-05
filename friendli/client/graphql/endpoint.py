# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Endpoint GQL Clients."""

from __future__ import annotations

from typing import Any, Dict, List

from friendli.client.graphql.base import GqlClient

CreateEndpointOp = """
mutation CreateEndpointWithHfModel($input: DedicatedEndpointCreateWithHfInput!) {
    dedicatedEndpointCreateWithHf(input: $input) {
        ... on UserPermissionError {
            message
        }
        ... on TeamNotExistError {
            message
        }
        ... on DedicatedEndpointCreateWithHfSuccess {
            endpoint {
                id
                name
                numGpu
                gpuType
                hfModelRepo
                status
                endpointUrl
                createdBy {
                    email
                    id
                    name
                }
                createdAt
                updatedAt
            }
        }
    }
}
"""

ListEndpointsOp = """
query ListDedicatedEndpointsInProject($id: ID!) {
    dedicatedProject(id: $id) {
        id
        name
        endpoints {
            totalCount
            edges {
                node {
                    id
                    name
                    hfModelRepo
                    gpuType
                    numGpu
                    hfModelRepo
                    status
                    phase {
                        ... on DedicatedEndpointPhaseRunning {
                            msg
                            desiredReplica
                            currReplica
                        }
                        ... on DedicatedEndpointPhaseInitializing {
                            step
                            msg
                        }
                    }
                    endpointUrl
                    createdBy {
                        email
                        id
                        name
                    }
                    advancedConfig {
                        maxBatchSize
                        autoscalingMin
                        autoscalingMax
                    }
                    createdAt
                    updatedAt
                }
            }
        }
    }
}
"""

GetEndpointOp = """
query GetDedicatedEndpoint($dedicatedEndpointId: ID!) {
    dedicatedEndpoint(id: $dedicatedEndpointId) {
        id
        name
        hfModelRepo
        gpuType
        numGpu
        status
        phase {
            ... on DedicatedEndpointPhaseInitializing {
                msg
                step
            }
            ... on DedicatedEndpointPhaseRunning {
                currReplica
                msg
                desiredReplica
            }
        }
        endpointUrl
        createdBy {
            email
            id
            name
        }
        createdAt
        updatedAt
        advancedConfig {
            autoscalingMax
            autoscalingMin
            maxBatchSize
        }
    }
}
"""

TerminateEndpointOp = """
mutation TerminateEndpoint($input: DedicatedEndpointTerminateInput!) {
    dedicatedEndpointTerminate(input: $input) {
        ... on DedicatedEndpointTerminateSuccess {
            endpoint {
                id
            }
        }
        ... on UserPermissionError {
            message
        }
        ... on TeamNotExistError {
            message
        }
    }
}
"""

ListInstancesOp = """
query DedicatedInstanceList {
    dedicatedInstanceList {
        id
        name
        disabledReason
        disabled
        options {
            quantity
            pricePerHour
            id
        }
    }
}
"""


class EndpointGqlClient(GqlClient):
    """Endpoint GQL client."""

    def create(  # pylint: disable=too-many-arguments
        self,
        team_id: str,
        project_id: str,
        name: str,
        model_repo: str,
        gpu_type: str,
        num_gpus: int,
    ) -> Dict[str, Any]:
        """Get current user info."""
        instance_option_id = self._get_instance_option_id(
            gpu_type=gpu_type, num_gpus=num_gpus
        )
        response = self.run(
            query=CreateEndpointOp,
            variables={
                "input": {
                    "teamId": team_id,
                    "projectId": project_id,
                    "name": name,
                    "modelRepo": model_repo,
                    "instanceOptionId": instance_option_id,
                }
            },
        )
        return response["dedicatedEndpointCreateWithHf"]["endpoint"]

    def list(self, project_id: str) -> List[Dict[str, Any]]:
        """List team projects."""
        response = self.run(query=ListEndpointsOp, variables={"id": project_id})
        return response["dedicatedProject"]["endpoints"]["edges"]

    def get(self, endpoint_id: str) -> Dict[str, Any]:
        """Get endpoint data."""
        response = self.run(
            query=GetEndpointOp,
            variables={"dedicatedEndpointId": endpoint_id},
        )
        return response["dedicatedEndpoint"]

    def terminate(self, team_id: str, project_id: str, endpoint_id: str) -> None:
        """Terminate a running endpoint."""
        self.run(
            query=TerminateEndpointOp,
            variables={
                "input": {
                    "teamId": team_id,
                    "projectId": project_id,
                    "endpointId": endpoint_id,
                }
            },
        )

    def _get_instance_option_id(self, gpu_type: str, num_gpus: int) -> str:
        response = self.run(query=ListInstancesOp)
        instances: list = response["dedicatedInstanceList"]

        available_gpus = []
        for instance in instances:
            name = instance["name"]
            if name == gpu_type:
                available_quantities = []
                for option in instance["options"]:
                    quantity = option["quantity"]
                    if quantity == num_gpus:
                        return option["id"]
                    available_quantities.append(quantity)

                raise ValueError(
                    f"'{gpu_type}' cannot be used with the quantity of '{num_gpus}'. "
                    f"Supported number of GPUs are {', '.join(available_quantities)}"
                )
            available_gpus.append(name)

        raise ValueError(
            f"GPU '{gpu_type}' is not supported. "
            f"Supported GPU types are {', '.join(name)}"
        )
