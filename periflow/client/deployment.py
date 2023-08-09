# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow DeploymentClient Service."""

from __future__ import annotations

from datetime import datetime
from string import Template
from typing import Any, Callable, Dict, List, Optional

from requests import Response

from periflow.client.base import Client, ProjectRequestMixin, safe_request


# TODO (ym): Replace this with periflow.utils.request.paginated_get after unifying schema
def paginated_get(
    response_getter: Callable[..., Response],
    path: Optional[str] = None,
    limit: int = 20,
    **params,
) -> List[Dict[str, Any]]:
    """Pagination listing."""
    page_size = min(10, limit)
    params = {"page_size": page_size, **params}
    response_dict = response_getter(path=path, params={**params}).json()
    items = response_dict["deployments"]
    next_cursor = response_dict["cursor"]

    while next_cursor is not None and len(items) < limit:
        response_dict = response_getter(
            path=path, params={**params, "cursor": next_cursor}
        ).json()
        items.extend(response_dict["deployments"])
        next_cursor = response_dict["cursor"]

    return items


class DeploymentClient(Client[str]):
    """Deployment client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_serving_uri("deployment/"))

    def get_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Get a deployment info."""
        response = safe_request(
            self.retrieve,
            err_prefix=f"Deployment ({deployment_id}) is not found. You may entered wrong ID.",
        )(pk=deployment_id)
        return response.json()

    def create_deployment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new deployment."""
        response = safe_request(self.post, err_prefix="Failed to post new deployment.")(
            json=config
        )
        return response.json()

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
        safe_request(
            self.partial_update,
            err_prefix=f"Failed to update scaler of deployment ({deployment_id}).",
        )(pk=deployment_id, path="scaler", json=json_body)

    def list_deployments(
        self, project_id: Optional[str], archived: bool, limit: int, from_oldest: bool
    ) -> List[Dict[str, Any]]:
        """List all deployments."""
        params: Dict[str, Any] = {"archived": archived}
        if project_id:
            params["project_id"] = project_id
        if from_oldest:
            params["descending"] = False

        return paginated_get(
            safe_request(self.list, err_prefix="Failed to list deployments."),
            limit=limit,
            **params,
        )

    def stop_deployment(self, deployment_id: str) -> None:
        """Delete a deployment."""
        safe_request(self.delete, err_prefix="Failed to delete deployment.")(
            pk=deployment_id
        )


class DeploymentLogClient(Client[str]):
    """Deployment log client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_serving_uri("deployment/$deployment_id/log/")
        )

    def get_deployment_logs(
        self, deployment_id: str, replica_index: int
    ) -> List[Dict[str, Any]]:
        """Get logs from a deployment."""
        response = safe_request(
            self.list,
            err_prefix=f"Log is not available for Deployment ({deployment_id})"
            f"with replica {replica_index}."
            "You may entered wrong ID or the replica is not running.",
        )(params={"replica_index": replica_index})
        return response.json()


class DeploymentMetricsClient(Client):
    """Deployment metrics client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_serving_uri("deployment/$deployment_id/metrics/")
        )

    def get_metrics(self, deployment_id: str, time_window: int) -> Dict[str, Any]:
        """Get metrics from a deployment."""
        response = safe_request(
            self.list,
            err_prefix=f"Deployment ({deployment_id}) is not found. You may entered wrong ID.",
        )(data=str(time_window))
        return response.json()


class DeploymentEventClient(Client):
    """Deployment event client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_serving_uri("deployment/$deployment_id/event/")
        )

    def get_events(self, deployment_id: str) -> List[Dict[str, Any]]:
        """Get deployment events."""
        response = safe_request(
            self.list,
            err_prefix=f"Events for deployment ({deployment_id}) is not found.",
        )()
        return response.json()


class DeploymentReqRespClient(Client):
    """Deployment request-response client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_serving_uri(
                "deployment/$deployment_id/req_resp/download/"
            )
        )

    def get_download_urls(
        self, deployment_id: str, start: datetime, end: datetime
    ) -> list[dict[str, str]]:
        """Get presigned URLs to download request-response logs."""
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        response = safe_request(
            self.list,
            err_prefix=f"Request-response logs for deployment({deployment_id}) are not found.",
        )(params=params)
        return response.json()


class PFSProjectUsageClient(Client[str], ProjectRequestMixin):
    """Project-level deployment usage client for serving."""

    def __init__(self, **kwargs):
        """Initialize project usage client for serving."""
        self.initialize_project()
        super().__init__(project_id=self.project_id, **kwargs)

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(
            self.url_provider.get_serving_uri("usage/project/$project_id/duration")
        )

    def get_usage(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Get deployment usage info."""
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        response = safe_request(
            self.list,
            err_prefix="Deployment usages are not found in the project.",
        )(params=params)
        return response.json()


class PFSVMClient(Client):
    """VM client for serving."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_serving_uri("vm/"))

    def list_vms(self) -> List[Dict[str, Any]]:
        """List all VM info."""
        response = safe_request(
            self.list,
            err_prefix="Cannot get available vm list from PFS server.",
        )()
        return response.json()
