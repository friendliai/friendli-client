# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Sync Model resource."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from httpx import Client, HTTPTransport
from rich.progress import Progress

from ....schema import FileDescriptorInput
from ....util.file_digest import file_sha256
from ....util.httpx.retry_transport import RetryTransportWrapper
from ...graphql.api import (
    AdapterModelCreateInput,
    AdapterPushCompleteResult,
    AdapterPushCompleteResultDedicatedModelPushAdapterCompleteDedicatedModelPushAdapterCompleteSuccess,
    AdapterPushCompleteVariables,
    AdapterPushStartResultDedicatedModelPushAdapterStartDedicatedModelPushAdapterStartSuccess,
    AdapterPushStartVariables,
    BaseModelCreateInput,
    BaseModelListResultDedicatedProjectModels,
    BaseModelListVariables,
    BasePushCompleteResult,
    BasePushCompleteVariables,
    BasePushStartVariables,
    BidirectionalConnectionInput,
    BigInt,
    ChunkGroupCommitVariables,
    ChunkGroupCreateVariables,
    ChunkPushCompleteVariables,
    ChunkPushStartVariables,
    DedicatedModelCommitChunkGroupInput,
    DedicatedModelCreateChunkGroupInput,
    DedicatedModelPushAdapterCompleteInput,
    DedicatedModelPushAdapterStartInput,
    DedicatedModelPushBaseCompleteInput,
    DedicatedModelPushBaseStartInput,
    DedicatedModelPushChunkCompleteInput,
    DedicatedModelPushChunkStartInput,
    DedicatedModelPushFileCompleteInput,
    DedicatedModelPushFileStartInput,
    FileChunkCompleteInput,
    FileChunkInput,
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

    def list(self, project_id: str) -> BaseModelListResultDedicatedProjectModels:
        """List all models in the specified project."""
        resp = self._sdk.gql_client.base_model_list(
            variables=BaseModelListVariables(
                dedicatedProjectId=project_id,
                conn=BidirectionalConnectionInput(
                    first=20,
                    skip=0,
                ),
            )
        )
        if (res := resp.dedicated_project) is None or (res.models is None):
            msg = f"No dedicated project found with ID {project_id}"
            raise ValueError(msg)

        return res.models

    def push_base_model(
        self,
        model_path: Path,
        project_id: str,
        model_name: str | None,
    ) -> BasePushCompleteResult:
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
        tokenizer_path = model_path / "tokenizer.json"
        tokenizer_config_path = model_path / "tokenizer_config.json"
        special_tokens_map_path = model_path / "special_tokens_map.json"
        safetensor_paths = [
            f for f in model_path.iterdir() if f.is_file() and "safetensor" in f.name
        ]
        if not config_path.exists():
            msg = "config.json not found"
            raise FileNotFoundError(msg)
        if not tokenizer_path.exists():
            msg = "tokenizer.json not found"
            raise FileNotFoundError(msg)
        if not safetensor_paths:
            msg = "No safetensor files found"
            raise FileNotFoundError(msg)

        print("Analyzing directory...")  # noqa: T201

        config_info = get_file_descriptor(config_path)
        tokenizer_info = get_file_descriptor(tokenizer_path)
        tokenizer_config_info = get_optional_file_descriptor(tokenizer_config_path)
        special_tokens_map_info = get_optional_file_descriptor(special_tokens_map_path)
        safetensor_infos = [get_file_descriptor(Path(f)) for f in safetensor_paths]

        # Calculate digest for each file and push them to the API

        model_structure = BaseModelCreateInput(
            config=FileDescriptorInputGql(
                digest=config_info.digest,
                filename=config_info.filename,
                size=BigInt(str(config_info.size)),
            ),
            tokenizer=FileDescriptorInputGql(
                digest=tokenizer_info.digest,
                filename=tokenizer_info.filename,
                size=BigInt(str(tokenizer_info.size)),
            ),
            tokenizerConfig=(
                FileDescriptorInputGql(
                    digest=tokenizer_config_info.digest,
                    filename=tokenizer_config_info.filename,
                    size=BigInt(str(tokenizer_config_info.size)),
                )
                if tokenizer_config_info
                else None
            ),
            specialTokensMap=(
                FileDescriptorInputGql(
                    digest=special_tokens_map_info.digest,
                    filename=special_tokens_map_info.filename,
                    size=BigInt(str(special_tokens_map_info.size)),
                )
                if special_tokens_map_info
                else None
            ),
            safetensors=[
                FileDescriptorInputGql(
                    digest=s.digest,
                    filename=s.filename,
                    size=BigInt(str(s.size)),
                )
                for s in safetensor_infos
            ],
        )

        start_vars = BasePushStartVariables(
            input=DedicatedModelPushBaseStartInput(
                projectId=project_id,
                name=model_name,
                modelStructure=model_structure,
            )
        )
        start_resp = self._sdk.gql_client.base_push_start(variables=start_vars)
        model_id = start_resp.dedicated_model_push_base_start.model.id

        # Push individual files
        self._push_file(config_info, model_id)
        self._push_file(tokenizer_info, model_id)
        self._push_file(tokenizer_config_info, model_id)
        self._push_file(special_tokens_map_info, model_id)
        for safetensor_info in safetensor_infos:
            self._push_file(safetensor_info, model_id)

        complete_vars = BasePushCompleteVariables(
            input=DedicatedModelPushBaseCompleteInput(
                modelId=model_id,
                modelStructure=model_structure,
            )
        )
        return self._sdk.gql_client.base_push_complete(variables=complete_vars)

    def push_adapter_model(
        self,
        model_path: Path,
        base_model_id: str,
        project_id: str,
        model_name: str | None,
    ) -> AdapterPushCompleteResult:
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
                size=BigInt(str(adapter_config_info.size)),
            ),
            safetensors=[
                FileDescriptorInputGql(
                    digest=s.digest,
                    filename=s.filename,
                    size=BigInt(str(s.size)),
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

        return complete_resp

    def _push_file(self, descriptor: FileDescriptorInput | None, model_id: str) -> None:
        if descriptor is None:
            return

        if descriptor.size > 100 * 1024 * 1024:  # 100 MB
            self._push_file_chunked(descriptor, model_id)
            return

        self._push_file_monolith(descriptor, model_id)

    def _push_file_chunked(
        self, descriptor: FileDescriptorInput, model_id: str
    ) -> None:
        print(f"Chunking & Pushing file {descriptor.filename}")  # noqa: T201

        create_vars = ChunkGroupCreateVariables(
            input=DedicatedModelCreateChunkGroupInput(
                modelId=model_id,
                fileInput=FileDescriptorInputGql(
                    digest=descriptor.digest,
                    filename=descriptor.filename,
                    size=BigInt(str(descriptor.size)),
                ),
            )
        )
        start_resp = self._sdk.gql_client.chunk_group_create(variables=create_vars)
        if (start_res := start_resp.dedicated_model_create_chunk_group) is None:
            return

        chunk_group_id = start_res.chunk_group_id

        # split blob_bytes into chunks, each 100MB
        # 100MB or 10 concurrent uploads
        chunk_size = max(1024 * 1024 * 100, descriptor.size // 200)
        chunk_size = min(chunk_size, 1024 * 1024 * 200)
        num_chunks = descriptor.size // chunk_size
        chunk_configs = [
            {
                "part_number": i + 1,
                "start": i * chunk_size,
                "end": (i + 1) * chunk_size,
                "size": chunk_size,
            }
            for i in range(num_chunks)
        ]
        # merge remaining chunk to last chunk
        chunk_configs[-1]["end"] = descriptor.size
        chunk_configs[-1]["size"] = (
            chunk_configs[-1]["end"] - chunk_configs[-1]["start"]
        )

        with ThreadPoolExecutor(max_workers=10) as executor, Progress() as progress:
            futures = {
                executor.submit(
                    self._push_chunk,
                    model_id,
                    chunk_group_id,
                    cc,
                    descriptor,
                ): cc
                for cc in chunk_configs
            }
            task = progress.add_task("[red]Uploading...", total=descriptor.size)
            for future in as_completed(futures):
                future.result()
                progress.update(task, advance=futures[future]["size"])

        self._sdk.gql_client.chunk_group_commit(
            variables=ChunkGroupCommitVariables(
                input=DedicatedModelCommitChunkGroupInput(
                    modelId=model_id,
                    chunkGroupId=chunk_group_id,
                    fileInput=FileDescriptorInputGql(
                        digest=descriptor.digest,
                        filename=descriptor.filename,
                        size=BigInt(str(descriptor.size)),
                    ),
                )
            )
        )

        print(f"Pushed file {descriptor.filename}")  # noqa: T201

    def _push_chunk(
        self,
        model_id: str,
        chunk_group_id: str,
        chunk_config: dict[str, int],
        descriptor: FileDescriptorInput,
    ) -> str:
        resp = self._sdk.gql_client.chunk_push_start(
            variables=ChunkPushStartVariables(
                input=DedicatedModelPushChunkStartInput(
                    modelId=model_id,
                    fileInput=FileChunkInput(
                        chunkGroupId=chunk_group_id,
                        partNumber=chunk_config["part_number"],
                        size=BigInt(str(chunk_config["size"])),
                    ),
                )
            )
        )
        if (res := resp.dedicated_model_push_chunk_start) is None:
            msg = "Chunk push start failed"
            raise RuntimeError(msg)

        url = res.upload_url  # type: ignore

        with descriptor.path.open("rb") as f:
            f.seek(chunk_config["start"])
            chunk = f.read(chunk_config["size"])

        transport = RetryTransportWrapper(
            HTTPTransport(
                http2=True,
                # TODO: limits=XX,
                # TODO: configure retries
                retries=4,
            )
        )

        with Client(timeout=300, transport=transport) as client:
            upload_resp = client.put(url=url, content=chunk, timeout=1200)
        upload_resp.raise_for_status()
        e_tag = upload_resp.headers["etag"]

        self._sdk.gql_client.chunk_push_complete(
            variables=ChunkPushCompleteVariables(
                input=DedicatedModelPushChunkCompleteInput(
                    modelId=model_id,
                    fileInput=FileChunkCompleteInput(
                        chunkGroupId=chunk_group_id,
                        partNumber=chunk_config["part_number"],
                        size=BigInt(str(chunk_config["size"])),
                        eTag=e_tag,
                    ),
                )
            )
        )

        return e_tag

    def _push_file_monolith(
        self, descriptor: FileDescriptorInput, model_id: str
    ) -> None:
        print(f"Pushing file {descriptor.filename}")  # noqa: T201

        start_vars = FilePushStartVariables(
            input=DedicatedModelPushFileStartInput(
                modelId=model_id,
                fileInput=FileDescriptorInputGql(
                    digest=descriptor.digest,
                    filename=descriptor.filename,
                    size=BigInt(str(descriptor.size)),
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
        with descriptor.path.open("rb") as f, Client(timeout=300) as client:
            files = {"file": f}
            upload_resp = client.post(
                upload_info.upload_url,
                data=upload_info.upload_body,
                files=files,
            )
        upload_resp.raise_for_status()

        complete_vars = FilePushCompleteVariables(
            input=DedicatedModelPushFileCompleteInput(
                modelId=model_id,
                fileInput=FileDescriptorInputGql(
                    digest=descriptor.digest,
                    filename=descriptor.filename,
                    size=BigInt(str(descriptor.size)),
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

        print(f"Pushed file {descriptor.filename}")  # noqa: T201


def get_file_descriptor(path: Path) -> FileDescriptorInput:
    """Get file descriptor."""
    digest = file_sha256(path)
    return FileDescriptorInput(
        digest=digest,
        filename=path.name,
        size=path.stat().st_size,
        path=path,
    )


def get_optional_file_descriptor(path: Path) -> FileDescriptorInput | None:
    """Get file descriptor."""
    if not path.exists():
        return None
    return get_file_descriptor(path)
