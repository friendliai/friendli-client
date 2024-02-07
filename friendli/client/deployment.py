# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Deployment Clients."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from friendli.client.base import Client, ProjectRequestMixin


class DeploymentClient(Client[str]):
    """Deployment client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_serving_uri("deployment/")

    def get_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Get a deployment info."""
        data = self.retrieve(
            pk=deployment_id,
        )
        return data

    def create_deployment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new deployment."""
        data = self.post(
            json=config,
        )
        return data

    def update_deployment_scaler(
        self, deployment_id: str, min_replicas: int, max_replicas: int
    ) -> None:
        """Update a deployment auto-scaler config."""
        json_body = {
            "scaler_config": {
                "min_replica_count": min_replicas,
                "max_replica_count": max_replicas,
            },
            "update_msg": f"Set min_replicas to {min_replicas}, max_replicas to {max_replicas}",
        }
        self.partial_update(
            pk=deployment_id,
            path="scaler",
            json=json_body,
        )

    def list_deployments(
        self, project_id: Optional[str], archived: bool, limit: int, from_oldest: bool
    ) -> List[Dict[str, Any]]:
        """List all deployments."""
        params: Dict[str, Any] = {"archived": archived}
        if project_id:
            params["project_id"] = project_id
        if from_oldest:
            params["descending"] = False

        deployments = self.list(
            pagination=True,
            limit=limit,
            params=params,
        )
        return deployments

    def stop_deployment(self, deployment_id: str) -> None:
        """Delete a deployment."""
        self.delete(
            pk=deployment_id,
        )


class DeploymentLogClient(Client[str]):
    """Deployment log client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_serving_uri("deployment/$deployment_id/log/")

    def get_deployment_logs(self, replica_index: int) -> List[Dict[str, Any]]:
        """Get logs from a deployment."""
        data = self.list(
            pagination=False,
            params={"replica_index": replica_index},
        )
        return data


class DeploymentMetricsClient(Client):
    """Deployment metrics client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_serving_uri("deployment/$deployment_id/metrics/")

    def get_metrics(
        self, start: datetime, end: datetime, time_window: int
    ) -> List[Dict[str, Any]]:
        """Get metrics from a deployment."""
        data = self.list(
            pagination=False,
            json={
                "start": start.isoformat(),
                "end": end.isoformat(),
                "time_window_sec": time_window,
                "metrics": ["error_rates", "p50_latency_ms", "p99_latency_ms"],
            },
        )
        return data


class DeploymentEventClient(Client):
    """Deployment event client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_serving_uri("deployment/$deployment_id/event/")

    def get_events(self) -> List[Dict[str, Any]]:
        """Get deployment events."""
        data = self.list(
            pagination=False,
        )
        return data


class DeploymentReqRespClient(Client):
    """Deployment request-response client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_serving_uri(
            "deployment/$deployment_id/req_resp/download/"
        )

    def get_download_urls(self, start: datetime, end: datetime) -> List[Dict[str, str]]:
        """Get presigned URLs to download request-response logs."""
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        data = self.list(
            pagination=False,
            params=params,
        )
        return data


class PFSProjectUsageClient(Client[str], ProjectRequestMixin):
    """Project-level deployment usage client for serving."""

    def __init__(self, **kwargs):
        """Initialize project usage client for serving."""
        self.initialize_project()
        super().__init__(project_id=self.project_id, **kwargs)

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_serving_uri("usage/project/$project_id/duration")

    def get_project_deployment_durations(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Get total deployment uptime info in the project."""
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        data = self.list(
            pagination=False,
            params=params,
        )
        return data


class PFSVMClient(Client):
    """VM client for serving."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_serving_uri("vm/")

    def list_vms(self) -> List[Dict[str, Any]]:
        """List all VM info."""
        data = self.list(
            pagination=False,
        )
        return data
