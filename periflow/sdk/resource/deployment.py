# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow deployment SDK."""

# pylint: disable=line-too-long, arguments-differ, too-many-arguments, too-many-locals, redefined-builtin

from __future__ import annotations

import os
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from uuid import UUID

import yaml
from dateutil.tz import tzlocal
from tqdm import tqdm

from periflow.client.deployment import (
    DeploymentClient,
    DeploymentEventClient,
    DeploymentLogClient,
    DeploymentMetricsClient,
    DeploymentReqRespClient,
    PFSProjectUsageClient,
)
from periflow.client.file import FileClient, GroupProjectFileClient
from periflow.client.group import GroupClient
from periflow.configurator.deployment import DRCConfigurator, OrcaDeploymentConfigurator
from periflow.context import get_current_group_id, get_current_project_id
from periflow.enums import (
    CloudType,
    DeploymentSecurityLevel,
    DeploymentType,
    GpuType,
    ServiceTier,
)
from periflow.errors import (
    AuthenticationError,
    EntityTooLargeError,
    InvalidConfigError,
    InvalidPathError,
    LowServicePlanError,
    NotFoundError,
)
from periflow.logging import logger
from periflow.schema.resource.v1.deployment import V1Deployment
from periflow.sdk.resource.base import ResourceAPI
from periflow.utils.format import extract_datetime_part, extract_deployment_id_part
from periflow.utils.fs import download_file, upload_file
from periflow.utils.maps import cloud_gpu_map, gpu_num_map
from periflow.utils.validate import validate_enums


class Deployment(ResourceAPI[V1Deployment, str]):
    """Deployment resource API."""

    @staticmethod
    def create(
        checkpoint_id: UUID,
        name: str,
        cloud: CloudType,
        region: str,
        gpu_type: GpuType,
        num_gpus: int,
        config: Dict[str, Any],
        deployment_type: DeploymentType = DeploymentType.PROD,
        description: Optional[str] = None,
        default_request_config: Optional[Dict[str, Any]] = None,
        security_level: DeploymentSecurityLevel = DeploymentSecurityLevel.PUBLIC,
        logging: bool = False,
        min_replicas: int = 1,
        max_replicas: int = 1,
    ) -> V1Deployment:
        """Creates a new deployment.

        Args:
            checkpoint_id (UUID): ID of checkpoint to deploy.
            name (str): The name of deployment.
            cloud (CloudType): Type of cloud provider.
            region (str): Cloud region to create a deployment.
            gpu_type (GpuType): Type of GPU.
            num_gpus (int): The number of GPUs.
            vm_type (VMType): Type of VM.
            config (Dict[str, Any]): Deployment configuration.
            deployment_type (DeploymentType, optional): Type of deployment. Defaults to DeploymentType.PROD.
            description (Optional[str], optional): Optional long description for the deployment. Defaults to None.
            default_request_config (Optional[Dict[str, Any]], optional): Default request configuration (e.g., stop words, bad words). Defaults to None.
            security_level (DeploymentSecurityLevel, optional): Security level of deployment endpoint. Defaults to DeploymentSecurityLevel.PUBLIC.
            logging (bool, optional): When True, enables request-response logging for the deployment if it is set. Defaults to False.
            min_replicas (int, optional): The number of minimum replicas. Defaults to 1.
            max_replicas (int, optional): The number of maximum replicas. Defaults to 1.

        Raises:
            AuthenticationError: Raised when project context is not configured.
            InvalidConfigError: Raised when any of the configurations (i.e., `config`, `default_request_config`, `min_replicas`, `max_replicas`) is invalid.
            EntityTooLargeError: Raised when the contents of `default_request_config` exceeds the 10GiB size limit.
            LowServicePlanError: Raised when the `deployment_type` is set to `Deployment.DEV` and service plan of the user's organization is not 'enterprise'.

        Examples:
            Basic usage:

            ```python
            import periflow as pf

            # Set up PeriFlow context.
            pf.init(
                api_key="YOUR_PERIFLOW_API_KEY",
                project_name="my-project",
            )

            # Create a deployment at GCP asia-northest3 region wtih one A100 GPU.
            config = {
                "max_batch_size": 256,
                "max_token_count": 8146,
                "max_num_tokens_to_replace": 0,
            }
            deployment = pf.Deployment.create(
                checkpoint_id="YOUR_CHECKPOINT_ID",
                name="my-deployment",
                cloud="gcp",
                region="asia-northeast3",
                gpu_type="a100",
                num_gpus=1,
                config=config,
            )
            ```

            The format of `config` should be:

            ```python
            {
                "max_batch_size": Optioanl[int],
                "max_token_count": Optioanl[int],
                "max_num_tokens_to_replace": Optional[int],
            }
            ```

            The format of `default_request_config` should be:

            ```python
            {
                "stop": Optional[List[str]],
                "stop_tokens": Optional[List[int]],
                "bad_words": Optional[List[str]],
                "bad_word_tokens": Optional[List[int]]
            }
            ```

            :::caution
            Note that `bad_words` and `bad_word_tokens` cannot be set at the same time. Similarly, `stop` and `stop_tokens` cannot be set at the same time.
            :::

            :::note
            When `min_replicas` and `max_replicas` are the same, deployment auto-scaling turns off.
            :::

        Returns:
            V1Deployment: The created deployment object.

        """
        # pylint: disable=too-many-statements
        cloud = validate_enums(cloud, CloudType)
        gpu_type = validate_enums(gpu_type, GpuType)
        deployment_type = validate_enums(deployment_type, DeploymentType)
        security_level = validate_enums(security_level, DeploymentSecurityLevel)

        org_id = get_current_group_id()
        if org_id is None:
            raise AuthenticationError(
                "Not authenticated. Please authenticate with 'pf init()' or 'pf login'."
            )
        project_id = get_current_project_id()
        if project_id is None:
            raise AuthenticationError(
                "Project context is not configured. "
                "Set the project context either by 'pf.init()' or 'pf project switch'."
            )

        group_client = GroupClient()
        if (
            group_client.get_group(pf_group_id=org_id)["plan"] == ServiceTier.BASIC
            and deployment_type == DeploymentType.DEV
        ):
            raise LowServicePlanError(
                "Deployment with the development type is only supported for the 'enterprise' plan."
            )

        if min_replicas > max_replicas:
            raise InvalidConfigError(
                f"Should be min_replicas('{min_replicas}') <= max_replicas('{max_replicas}')."
            )

        if gpu_type not in cloud_gpu_map[cloud]:
            raise InvalidConfigError(
                f"GPU type {gpu_type.value} is not supported in cloud {cloud.value}."
            )

        if num_gpus not in gpu_num_map[gpu_type]:
            raise InvalidConfigError(
                f"Num gpus {num_gpus} is not supported for GPU {gpu_type.value}."
            )

        deploy_configurator = OrcaDeploymentConfigurator(config=config)
        deploy_configurator.validate()
        config = {"orca_config": config}

        if default_request_config is not None:
            drc_configurator = DRCConfigurator(config=default_request_config)
            drc_configurator.validate()

            file_client = FileClient()
            group_file_client = GroupProjectFileClient()

            with TemporaryDirectory() as dir:
                drc_file_name = "drc.yaml"
                drc_file_path = os.path.join(dir, drc_file_name)
                with open(drc_file_path, "w", encoding="utf-8") as file:
                    yaml.dump(default_request_config, file)

                file_size = os.stat(drc_file_path).st_size
                if file_size > 10737418240:  # 10GiB
                    raise EntityTooLargeError(
                        "The default request config size should be smaller than 10GiB."
                    )

                file_info = {
                    "name": drc_file_name,
                    "path": drc_file_name,
                    "mtime": datetime.fromtimestamp(
                        os.stat(drc_file_path).st_mtime, tz=tzlocal()
                    ).isoformat(),
                    "size": file_size,
                }
                file_id = group_file_client.create_misc_file(file_info=file_info)["id"]

                upload_url = file_client.get_misc_file_upload_url(misc_file_id=file_id)
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Uploading default request config",
                ) as t:
                    upload_file(
                        file_path=drc_file_path,
                        url=upload_url,
                        ctx=t,
                    )
                file_client.make_misc_file_uploaded(misc_file_id=file_id)
                config["orca_config"]["default_request_config_id"] = file_id

        config["orca_config"]["num_devices"] = num_gpus

        config["scaler_config"] = {}
        config["scaler_config"]["min_replica_count"] = min_replicas
        config["scaler_config"]["max_replica_count"] = max_replicas

        request_data = {
            "project_id": str(project_id),
            "model_id": str(checkpoint_id),
            "deployment_type": deployment_type.value,
            "name": name,
            "vm": {"gpu_type": gpu_type.value},
            "cloud": cloud.value,
            "region": region,
            "total_gpus": num_gpus,
            "infrequest_perm_check": security_level
            == DeploymentSecurityLevel.PROTECTED,
            "infrequest_log": logging,
            **config,
        }
        if description is not None:
            request_data["description"] = description

        client = DeploymentClient()
        deployment_raw = client.create_deployment(request_data)
        deployment = V1Deployment.model_validate(deployment_raw)
        return deployment

    @staticmethod
    def list(
        limit: int = 20,
        include_terminated: bool = False,
        from_oldest: bool = False,
        all_in_org: bool = False,
    ) -> List[V1Deployment]:
        """Lists deployments.

        Args:
            limit (int, optional): The maximum number of retrieved results. Defaults to 20.
            include_terminated (bool, optional): When True, includes the terminated deployments in the results. Defaults to False.
            from_oldest (bool, optional): List from the oldest deployment. Defaults to False.

        Raises:
            AuthenticationError: Raised when project context is not configured.

        Returns:
            List[V1Deployment]: Retrieved deployments.

        """
        if all_in_org:
            prj_id = None
        else:
            project_id = get_current_project_id()
            if project_id is None:
                raise AuthenticationError(
                    "Project context is not configured. "
                    "Set the project context either by 'pf.init' or 'pf project switch'."
                )
            prj_id = str(project_id)

        client = DeploymentClient()
        deployments = [
            V1Deployment.model_validate(deployment_raw)
            for deployment_raw in client.list_deployments(
                project_id=prj_id,
                archived=False,
                limit=limit,
                from_oldest=from_oldest,
            )
        ]
        num_active_deployments = len(deployments)
        if include_terminated and limit > num_active_deployments:
            deployments += [
                V1Deployment.model_validate(deployment_raw)
                for deployment_raw in client.list_deployments(
                    project_id=prj_id,
                    archived=True,
                    limit=limit - num_active_deployments,
                    from_oldest=from_oldest,
                )
            ]

        return deployments

    @staticmethod
    def get(id: str, *args, **kwargs) -> V1Deployment:
        """Gets deployment info.

        Args:
            id (str): ID of deployment to retrieve.

        Returns:
            V1Deployment: Retrieved deployment object.

        """
        client = DeploymentClient()
        deployment_raw = client.get_deployment(id)
        deployment = V1Deployment.model_validate(deployment_raw)

        return deployment

    @staticmethod
    def stop(id: str) -> None:
        """Stops a running deployment.

        Args:
            id (str): ID of deployment to stop.

        """
        client = DeploymentClient()
        client.stop_deployment(id)

    @staticmethod
    def get_metrics(id: str, time_window: int = 60) -> Dict[str, Any]:
        """Gets metrics of a deployment.

        Args:
            id (str): ID of deployment to get metrics.
            time_window (int, optional): Time window of results in seconds. Defaults to 60.

        Returns:
            Dict[str, Any]: Retrieved metrics data.

        """
        metrics_client = DeploymentMetricsClient(deployment_id=id)
        return metrics_client.get_metrics(deployment_id=id, time_window=time_window)

    @staticmethod
    def get_usage(since: datetime, until: datetime) -> Dict[str, Any]:
        """Gets usage info of a deployment.

        Args:
            since (datetime): Start datetime of the deployment usages to fetch.
            until (datetime): End datetime of the deployment usages to fetch.

        Returns:
            Dict[str, Any]: Retrieved deployment usage info.

        """
        client = PFSProjectUsageClient()
        return client.get_usage(since, until)

    @staticmethod
    def get_logs(id: str, replica_index: int = 0) -> List[Dict[str, Any]]:
        """Gets logs from a deployment.

        Args:
            id (str): ID of deployment to get logs.
            replica_index (int, optional): Index of deployment replica to retrieve logs. Defaults to 0.

        Returns:
            List[Dict[str, Any]]: Retrieved log data.

        """
        client = DeploymentLogClient(deployment_id=id)
        return client.get_deployment_logs(deployment_id=id, replica_index=replica_index)

    @staticmethod
    def adjust_replica_config(id: str, min_replicas: int, max_replicas: int) -> None:
        """Adjusts replica configuration of a running deployment.

        Args:
            id (str): ID of deployment to adjust the replica configuration.
            min_replicas (int): Minimum replica count.
            max_replicas (int): Maximum replica count.

        Raises:
            InvalidConfigError: Raised when any of the configurations (i.e., `min_replicas`, `max_replicas`) is invalid.

        """
        if min_replicas > max_replicas:
            raise InvalidConfigError(
                f"Should be min_replicas('{min_replicas}') <= max_replicas('{max_replicas}')."
            )
        client = DeploymentClient()
        client.update_deployment_scaler(
            deployment_id=id,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )

    @staticmethod
    def get_events(id: str) -> List[Dict[str, Any]]:
        """Gets events from a deployment.

        Args:
            id (str): ID of deployment to get events.

        Returns:
            List[Dict[str, Any]]: Deployment event info.

        """
        client = DeploymentEventClient(deployment_id=id)
        return client.get_events(deployment_id=id)

    @staticmethod
    def download_req_resp_logs(
        id: str, since: datetime, until: datetime, save_dir: Optional[str] = None
    ) -> None:
        """Downloads a file that has request-response logs.

        Args:
            id (str): ID of deployment.
            since (datetime): Start time of request-response logs.
            until (datetime): End time of request-response logs.
            save_dir (Optional[str], optional): Path to save logs. Set to the current directory if it is `None`. Defaults to None.

        Raises:
            InvalidPathError: Raised when `save_dir` does not exist or is read-only.
            InvalidConfigError: Raised when `since` <= `until` is not satisfied.
            NotFoundError: Raised when the deployment request-response logs are not found for the given time range.

        """
        if save_dir is not None and not os.path.isdir(save_dir):
            raise InvalidPathError(f"Directory '{save_dir}' is not found.")
        save_dir = save_dir or os.getcwd()

        if not os.access(save_dir, os.W_OK):
            raise InvalidPathError(f"Cannot save logs to {save_dir} which is readonly.")

        if since > until:
            raise InvalidConfigError(
                "Time value of `since` should be earlier than the value of `until`."
            )

        client = DeploymentReqRespClient(deployment_id=id)
        download_infos = client.get_download_urls(
            deployment_id=id, start=since, end=until
        )
        if len(download_infos) == 0:
            raise NotFoundError(f"No log exists for the deployment '{id}'.")

        for i, download_info in enumerate(download_infos):
            logger.info("Downloading files %d/%d...", i + 1, len(download_infos))
            full_storage_path = download_info["path"]
            deployment_id_part = extract_deployment_id_part(full_storage_path)
            timestamp_part = extract_datetime_part(full_storage_path)
            filename = f"{deployment_id_part}_{timestamp_part}.log"
            download_file(
                url=download_info["url"], out=os.path.join(save_dir, filename)
            )
