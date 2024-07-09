# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Sync Model resource."""

from __future__ import annotations

from pathlib import Path

from httpx import Client

from ....schema import FileDescriptorInput
from ....util.file_digest import file_sha256
from ...graphql.api import (
    AdapterModelCreateInput,
    AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccess,
    AdapterPushCompleteVariables,
    AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccess,
    AdapterPushStartVariables,
    DedicatedModelPushAdapterCompleteInput,
    DedicatedModelPushAdapterStartInput,
    DedicatedModelPushFileCompleteInput,
    DedicatedModelPushFileStartInput,
    FilePushCompleteResultDedicatedModelPushFileCompleteDedicatedModelPushFileCompleteSuccess,
    FilePushCompleteVariables,
    FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartAlreadyExistError,
    FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccess,
    FilePushStartVariables,
)
from ...graphql.api import (
    FileDescriptorInput as FileDescriptorInputGql,
)
from ._base import ResourceBase


class ModelResource(ResourceBase):
    """Model resource for Friendli Suite API."""

    def push_base_model(
        self,
        model_path: Path,
        project_id: str,
        model_name: str | None,
    ) -> None:
        """Start procedure for uploading base model.

        It first checks for the provided directory structure, then, calculates the
        digest of each files.

        Args:
            model_path (Path): Path to the directory containing the base model files.
            project_id (str, optional): ID of the project to upload the model to.
            model_name (str, optional): Name of the model. Auto-generated if not
                provided.

        """
        # Check directory against file pattern
        # Check for directory structure
        if not model_path.exists():
            msg = f"Model directory {model_path} does not exist"
            raise FileNotFoundError(msg)

        config_path = model_path / "config.json"
        safetensor_paths = [
            f for f in model_path.iterdir() if f.is_file() and "safetensor" in f.name
        ]
        if not config_path.exists():
            msg = "adapter_config.json not found"
            raise FileNotFoundError(msg)
        if not safetensor_paths:
            msg = "No safetensor files found"
            raise FileNotFoundError(msg)

        config_info = get_file_descriptor(config_path)
        safetensor_infos = [get_file_descriptor(Path(f)) for f in safetensor_paths]

        # Calculate digest for each file and push them to the API

        _ = config_info, safetensor_infos, project_id, model_name

        # Request

    def push_adapter_model(
        self,
        model_path: Path,
        base_model_id: str,
        project_id: str,
        model_name: str | None,
    ) -> None:
        """Upload base model.

        It first checks for the provided directory structure, then, calculates the
        digest of each files.

        Args:
            model_path (Path): Path to the directory containing the base model files.
            base_model_id (str): ID of the base model to push the adapter model on.
            project_id (str, optional): ID of the project to upload the model to.
            model_name (str, optional): Name of the model. Auto-generated if not
                provided.

        """
        # Check for directory structure
        if not model_path.exists():
            msg = f"Model directory {model_path} does not exist"
            raise FileNotFoundError(msg)

        adapter_config_path = model_path / "adapter_config.json"
        safetensor_paths = [
            f for f in model_path.iterdir() if f.is_file() and "safetensor" in f.name
        ]
        if not adapter_config_path.exists():
            msg = "adapter_config.json not found"
            raise FileNotFoundError(msg)
        if not safetensor_paths:
            msg = "No safetensor files found"
            raise FileNotFoundError(msg)

        adapter_config_info = get_file_descriptor(adapter_config_path)
        safetensor_infos = [get_file_descriptor(Path(f)) for f in safetensor_paths]

        model_structure = AdapterModelCreateInput(
            adapterConfig=FileDescriptorInputGql(
                digest=adapter_config_info.digest,
                filename=adapter_config_info.filename,
                size=adapter_config_info.size,
            ),
            safetensors=[
                FileDescriptorInputGql(
                    digest=s.digest,
                    filename=s.filename,
                    size=s.size,
                )
                for s in safetensor_infos
            ],
        )

        start_vars = AdapterPushStartVariables(
            input=DedicatedModelPushAdapterStartInput(
                projectId=project_id,
                name=model_name,
                baseModelId=base_model_id,
                modelStructure=model_structure,
            )
        )
        start_resp = self._sdk.gql_client.adapter_push_start(variables=start_vars)
        if not isinstance(
            (start_res := start_resp.dedicated_model_push_adapter_start),
            AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccess,
        ):
            msg = f"Adapter push start failed: {start_res.message}"
            raise RuntimeError(msg)  # noqa: TRY004

        model_id = start_res.adapter.id

        # Push individual files
        self._push_file(adapter_config_info, model_id)
        for safetensor_info in safetensor_infos:
            self._push_file(safetensor_info, model_id)

        complete_vars = AdapterPushCompleteVariables(
            input=DedicatedModelPushAdapterCompleteInput(
                adapterId=model_id,
                modelStructure=model_structure,
            )
        )
        complete_resp = self._sdk.gql_client.adapter_push_complete(
            variables=complete_vars
        )
        if not isinstance(
            (complete_res := complete_resp.dedicated_model_push_adapter_complete),
            AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccess,
        ):
            msg = f"Adapter push complete failed: {complete_res.message}"
            raise RuntimeError(msg)  # noqa: TRY004

    def _push_file(self, descriptor: FileDescriptorInput, model_id: str) -> None:
        start_vars = FilePushStartVariables(
            input=DedicatedModelPushFileStartInput(
                modelId=model_id,
                fileInput=FileDescriptorInputGql(
                    digest=descriptor.digest,
                    filename=descriptor.filename,
                    size=descriptor.size,
                ),
            )
        )
        start_resp = self._sdk.gql_client.file_push_start(variables=start_vars)
        if isinstance(
            (start_res := start_resp.dedicated_model_push_file_start),
            FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartAlreadyExistError,
        ):
            return

        if not isinstance(
            start_res,
            FilePushStartResultDedicatedModelPushFileStartDedicatedModelPushFileStartSuccess,
        ):
            msg = f"File push start failed: {start_res.message}"
            raise RuntimeError(msg)  # noqa: TRY004

        upload_info = start_res.upload_info
        with descriptor.path.open("rb") as f, Client(timeout=180) as client:
            files = {"file": f}
            upload_resp = client.post(
                upload_info.upload_url,
                data=upload_info.upload_body,
                files=files,
            )
        upload_resp.raise_for_status()

        print(f"Pushing file {descriptor.filename}")  # noqa: T201

        complete_vars = FilePushCompleteVariables(
            input=DedicatedModelPushFileCompleteInput(
                modelId=model_id,
                fileInput=FileDescriptorInputGql(
                    digest=descriptor.digest,
                    filename=descriptor.filename,
                    size=descriptor.size,
                ),
            )
        )
        complete_resp = self._sdk.gql_client.file_push_complete(variables=complete_vars)
        if not isinstance(
            (complete_res := complete_resp.dedicated_model_push_file_complete),
            FilePushCompleteResultDedicatedModelPushFileCompleteDedicatedModelPushFileCompleteSuccess,
        ):
            msg = f"File push complete failed: {complete_res.message}"
            raise RuntimeError(msg)  # noqa: TRY004


def get_file_descriptor(path: Path) -> FileDescriptorInput:
    """Get file descriptor."""
    digest = file_sha256(path)
    return FileDescriptorInput(
        digest=digest,
        filename=path.name,
        size=path.stat().st_size,
        path=path,
    )
