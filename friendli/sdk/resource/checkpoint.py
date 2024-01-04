# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Checkpoint SDK."""

# pylint: disable=line-too-long, arguments-differ, too-many-arguments, too-many-statements, too-many-locals, redefined-builtin, too-many-lines

from __future__ import annotations

import functools
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import yaml

from friendli.client.catalog import CatalogClient
from friendli.client.checkpoint import (
    CheckpointClient,
    CheckpointFormClient,
    GroupProjectCheckpointClient,
)
from friendli.client.credential import CredentialClient
from friendli.cloud.storage import build_storage_client
from friendli.enums import (
    CatalogImportMethod,
    CheckpointCategory,
    CheckpointStatus,
    CredType,
    StorageType,
)
from friendli.errors import FriendliInternalError, InvalidConfigError, NotFoundError
from friendli.logging import logger
from friendli.schema.resource.v1.checkpoint import V1Checkpoint
from friendli.schema.resource.v1.transfer import MultipartUploadTask, UploadTask
from friendli.sdk.resource.base import ResourceAPI
from friendli.utils.fs import (
    attach_storage_path_prefix,
    get_file_info,
    strip_storage_path_prefix,
)
from friendli.utils.maps import cred_type_map, cred_type_map_inv
from friendli.utils.transfer import (
    ChunksizeAdjuster,
    DeferQueue,
    DownloadManager,
    UploadManager,
)
from friendli.utils.validate import (
    validate_checkpoint_attributes,
    validate_cloud_storage_type,
    validate_enums,
    validate_storage_region,
)


class Checkpoint(ResourceAPI[V1Checkpoint, UUID]):
    """Checkpoint resource API."""

    def create(
        self,
        *,
        name: str,
        credential_id: UUID,
        cloud_storage: StorageType,
        region: str,
        storage_name: str,
        storage_path: Optional[str] = None,
        iteration: Optional[int] = None,
        attr_file_path: Optional[str] = None,
    ) -> V1Checkpoint:
        """Creates a checkpoint by linking the existing cloud storage (e.g., AWS S3, GCS, Azure Blob Storage) with Friendli.

        Args:
            name (str): The name of checkpoint to create.
            credential_id (UUID): Credential ID to access the cloud storage.
            cloud_storage (StorageType): Cloud storage type.
            region (str): Cloud region.
            storage_name (str): Storage name (e.g., AWS S3 bucket name).
            storage_path (Optional[str], optional): Path to the storage object (e.g., AWS S3 bucket key). Defaults to None.
            iteration (Optional[int], optional): The iteration of the checkpoint. Defaults to None.
            attr_file_path (Optional[str], optional): Path to the checkpoint attribute YAML file. Defaults to None.

        Raises:
            InvalidConfigError: Raised when checkpoint attribute file located at `attr_file_path` has invalid YAML format. Also raised when the credential with `credential_id` is not for the cloud provider of `cloud_storage`. Also raised when `region` is invalid.
            NotSupportedError: Raised when `cloud_storage` is not supported yet.
            InvalidAttributesError: Raised when the checkpoint attributes described in `attr_file_path` is in the invalid format.

        Returns:
            V1Checkpoint: Created checkpoint object.

        Examples:
            Basic usage:

            ```python
            from friendli import FriendliResource

            # Set up Friendli context.
            client = FriendliResource(
                token="YOUR_FRIENDLI_TOKEN",
                project_name="my-project",
            )

            # Create a checkpoint by linking an existing S3 bucket.
            checkpoint = client.checkpoint.create(
                name="my-checkpoint",
                credential_id="YOUR_CREDENTIAL_ID",
                cloud_stroage="s3",
                region="us-east-1",
                storage_name="my-bucket",
                storage_path="path/to/ckpt",
                attr_file_path="path/to/attr.yaml",
            )
            ```

            :::info
            You need to create a credential to access the S3 storage in advance.
            :::

            :::note
            An example of attribute files is as follows.
            You have the flexibility to modify the value of each field according to your preferences,
            but it is mandatory to fill in every field.

            ```yaml
            # blenderbot
            model_type: blenderbot
            dtype: fp16
            head_size: 80
            num_heads: 32
            hidden_size: 2560
            ff_intermediate_size: 10240
            num_encoder_layers: 2
            num_decoder_layers: 24
            max_input_length: 128
            max_output_length: 128
            vocab_size: 8008
            eos_token: 2
            decoder_start_token: 1

            # bloom
            model_type: bloom
            dtype: fp16
            head_size: 128
            num_heads: 32
            num_layers: 30
            max_length: 2048
            vocab_size: 250880
            eos_token: 2

            # gpt
            model_type: gpt
            dtype: fp16
            head_size: 64
            num_heads: 25
            num_layers: 48
            max_length: 1024
            vocab_size: 50257
            eos_token: 50256

            # gpt-j
            model_type: gpt-j
            dtype: fp16
            head_size: 256
            rotary_dim: 64
            num_heads: 16
            num_layers: 28
            max_length: 2048
            vocab_size: 50400
            eos_token: 50256

            # gpt-neox
            model_type: gpt-neox
            dtype: fp16
            head_size: 128
            rotary_dim: 32
            num_heads: 40
            num_layers: 36
            max_length: 2048
            vocab_size: 50280
            eos_token: 0

            # llama
            model_type: llama
            dtype: fp16
            head_size: 128
            rotary_dim: 128
            num_heads: 32
            num_kv_heads: 32
            num_layers: 32
            ff_intermediate_size: 11008
            max_length: 2048
            vocab_size: 32000
            eos_token: 1

            # opt
            model_type: opt
            dtype: fp16
            head_size: 128
            num_heads: 32
            num_layers: 32
            max_length: 2048
            vocab_size: 50272
            eos_token: 2

            # t5
            model_type: t5
            dtype: fp16
            head_size: 128
            num_heads: 32
            hidden_size: 1024
            ff_intermediate_size: 16384
            num_encoder_layers: 24
            num_decoder_layers: 24
            max_input_length: 512
            max_output_length: 512
            num_pos_emb_buckets: 32
            max_pos_distance: 128
            vocab_size: 32100
            eos_token: 1
            decoder_start_token: 0

            # t5-v1_1
            model_type: t5-v1_1
            dtype: fp16
            head_size: 64
            num_heads: 32
            hidden_size: 2048
            ff_intermediate_size: 5120
            num_encoder_layers: 24
            num_decoder_layers: 24
            max_input_length: 512
            max_output_length: 512
            num_pos_emb_buckets: 32
            max_pos_distance: 128
            vocab_size: 32128
            eos_token: 1
            decoder_start_token: 0
            ```
            :::

        """
        cloud_storage = validate_enums(cloud_storage, StorageType)
        validate_cloud_storage_type(cloud_storage)
        validate_storage_region(vendor=cloud_storage, region=region)

        attr = {}
        if attr_file_path is not None:
            try:
                with open(attr_file_path, "r", encoding="utf-8") as attr_f:
                    attr = yaml.safe_load(attr_f)
            except yaml.YAMLError as exc:
                raise InvalidConfigError(
                    f"The attribute YAML file has invalid format: {str(exc)}"
                ) from exc

        if attr:
            validate_checkpoint_attributes(attr)

        credential_client = CredentialClient()
        credential = credential_client.get_credential(credential_id)
        if credential["type"] != cred_type_map[CredType(cloud_storage.value)]:
            raise InvalidConfigError(
                "Credential type and cloud vendor mismatch: "
                f"{cred_type_map_inv[credential['type']]} and {cloud_storage.value}."
            )

        storage_helper = build_storage_client(
            cloud_storage, credential_json=credential["value"]
        )
        if storage_path is not None:
            storage_path = storage_path.strip("/")
        files = storage_helper.list_storage_files(storage_name, storage_path)
        if storage_path is not None:
            storage_name = f"{storage_name}/{storage_path}"

        group_ckpt_client = GroupProjectCheckpointClient()
        dist_config = {
            "pp_degree": 1,
            "dp_degree": 1,
            "mp_degree": 1,
            "dp_mode": "allreduce",
            "parallelism_order": ["pp", "dp", "mp"],
        }
        raw_ckpt = group_ckpt_client.create_checkpoint(
            name=name,
            vendor=cloud_storage,
            region=region,
            credential_id=credential_id,
            iteration=iteration,
            storage_name=storage_name,
            files=files,
            dist_config=dist_config,
            attributes=attr,
        )
        ckpt = V1Checkpoint.model_validate(raw_ckpt)

        ckpt_client = CheckpointClient()
        raw_ckpt = ckpt_client.activate_checkpoint(ckpt.id)
        ckpt = V1Checkpoint.model_validate(raw_ckpt)
        if ckpt.forms:
            for file in ckpt.forms[0].files:
                file.path = strip_storage_path_prefix(file.path)
        return ckpt

    def list(
        self,
        *,
        category: Optional[CheckpointCategory] = None,
        limit: int = 20,
        deleted: bool = False,
    ) -> List[V1Checkpoint]:
        """Lists checkpoints.

        Args:
            category (Optional[CheckpointCategory], optional): Filters by category. Defaults to None.
            limit (int, optional): The maximum number of retrieved results. Defaults to 20.
            deleted (bool, optional): Filters only the deleted checkpoints. Defaults to False.

        Returns:
            List[V1Checkpoint]: A list of retrieved checkpoints.

        Examples:
            To get the latest 100 checkpoints:

            ```python
            from friendli import FriendliResource

            # Set up Friendli context.
            client = FriendliResource(
                token="YOUR_FRIENDLI_TOKEN",
                project_name="my-project",
            )

            checkpoints = client.checkpoint.list(limit=100)
            ```

            To get the deleted checkpoints created by users:

            ```python
            checkpoints = client.checkpoint.list(
                category="USER", deleted=True
            )
            ```

        """
        if category is not None:
            category = validate_enums(category, CheckpointCategory)

        client = GroupProjectCheckpointClient()
        checkpoints = [
            V1Checkpoint.model_validate(raw_ckpt)
            for raw_ckpt in client.list_checkpoints(
                category, limit=limit, deleted=deleted
            )
        ]
        return checkpoints

    def get(self, id: UUID, *args, **kwargs) -> V1Checkpoint:
        """Gets a specific checkpoint.

        Args:
            id (UUID): ID of checkpoint to retrieve.

        Returns:
            V1Checkpoint: The retrieved checkpoint object.

        Examples:
            Basic usage:

            ```python
            from friendli import FriendliResource

            # Set up Friendli context.
            client = FriendliResource(
                token="YOUR_FRIENDLI_TOKEN",
                project_name="my-project",
            )

            checkpoint = client.checkpoint.get(id="YOUR_CHECKPOINT_ID")
            ```

        """
        client = CheckpointClient()
        raw_ckpt = client.get_checkpoint(id)
        ckpt = V1Checkpoint.model_validate(raw_ckpt)
        if ckpt.forms:
            for file in ckpt.forms[0].files:
                file.path = strip_storage_path_prefix(file.path)
        return ckpt

    def delete(self, id: UUID) -> None:
        """Deletes a checkpoint.

        Args:
            id (UUID): ID of checkpoint to delete.

        Examples:
            Basic usage:

            ```python
            from friendli import FriendliResource

            # Set up Friendli context.
            client = FriendliResource(
                token="YOUR_FRIENDLI_TOKEN",
                project_name="my-project",
            )

            client.checkpoint.delete(id="YOUR_CHECKPOINT_ID")
            ```

        """
        client = CheckpointClient()
        client.delete_checkpoint(id)

    def upload(
        self,
        *,
        name: str,
        source_path: str,
        iteration: Optional[int] = None,
        attr_file_path: Optional[str] = None,
        max_workers: int = min(32, (os.cpu_count() or 1) + 4),
    ) -> V1Checkpoint:
        """Creates a new checkpoint by uploading files in the local file system.

        Args:
            name (str): The name of checkpoint to create.
            source_path (str): Local path to the source file or directory to upload.
            iteration (Optional[int], optional): Trained step of the checkpoint. Defaults to None.
            attr_file_path (Optional[str], optional): Path to the checkpoint attribute YAML file. Defaults to None.
            max_workers (int, optional): The number of concurrency. Defaults to min(32, (os.cpu_count() or 1) + 4).

        Raises:
            NotFoundError: Raised when `source_path` does not exist.
            InvalidConfigError: Raised when the attribute file located at `attr_file_path` has invalid YAML format.
            InvalidAttributesError: Raised when the checkpoint attributes described in `attr_file_path` is in the invalid format.

        Returns:
            V1Checkpoint: Created checkpoint object.

        Examples:
            Basic usage:

            ```python
            from friendli import FriendliResource

            # Set up Friendli context.
            client = FriendliResource(
                token="YOUR_FRIENDLI_TOKEN",
                project_name="my-project",
            )

            # Create a checkpoint by loading files located at "local/path/to/ckpt" directory.
            checkpoint = friendli.checkpoint.upload(
                name="my-checkpoint",
                source_path="local/path/to/ckpt",
                attr_file_path="local/path/to/attr.yaml",
            )
            ```

            :::note
            An example of attribute files is as follows.
            You have the flexibility to modify the value of each field according to your preferences,
            but it is mandatory to fill in every field.

            ```yaml
            # blenderbot
            model_type: blenderbot
            dtype: fp16
            head_size: 80
            num_heads: 32
            hidden_size: 2560
            ff_intermediate_size: 10240
            num_encoder_layers: 2
            num_decoder_layers: 24
            max_input_length: 128
            max_output_length: 128
            vocab_size: 8008
            eos_token: 2
            decoder_start_token: 1

            # bloom
            model_type: bloom
            dtype: fp16
            head_size: 128
            num_heads: 32
            num_layers: 30
            max_length: 2048
            vocab_size: 250880
            eos_token: 2

            # gpt
            model_type: gpt
            dtype: fp16
            head_size: 64
            num_heads: 25
            num_layers: 48
            max_length: 1024
            vocab_size: 50257
            eos_token: 50256

            # gpt-j
            model_type: gpt-j
            dtype: fp16
            head_size: 256
            rotary_dim: 64
            num_heads: 16
            num_layers: 28
            max_length: 2048
            vocab_size: 50400
            eos_token: 50256

            # gpt-neox
            model_type: gpt-neox
            dtype: fp16
            head_size: 128
            rotary_dim: 32
            num_heads: 40
            num_layers: 36
            max_length: 2048
            vocab_size: 50280
            eos_token: 0

            # llama
            model_type: llama
            dtype: fp16
            head_size: 128
            rotary_dim: 128
            num_heads: 32
            num_kv_heads: 32
            num_layers: 32
            ff_intermediate_size: 11008
            max_length: 2048
            vocab_size: 32000
            eos_token: 1

            # opt
            model_type: opt
            dtype: fp16
            head_size: 128
            num_heads: 32
            num_layers: 32
            max_length: 2048
            vocab_size: 50272
            eos_token: 2

            # t5
            model_type: t5
            dtype: fp16
            head_size: 128
            num_heads: 32
            hidden_size: 1024
            ff_intermediate_size: 16384
            num_encoder_layers: 24
            num_decoder_layers: 24
            max_input_length: 512
            max_output_length: 512
            num_pos_emb_buckets: 32
            max_pos_distance: 128
            vocab_size: 32100
            eos_token: 1
            decoder_start_token: 0

            # t5-v1_1
            model_type: t5-v1_1
            dtype: fp16
            head_size: 64
            num_heads: 32
            hidden_size: 2048
            ff_intermediate_size: 5120
            num_encoder_layers: 24
            num_decoder_layers: 24
            max_input_length: 512
            max_output_length: 512
            num_pos_emb_buckets: 32
            max_pos_distance: 128
            vocab_size: 32128
            eos_token: 1
            decoder_start_token: 0
            ```
            :::

        """
        expand = source_path.endswith("/") or os.path.isfile(source_path)
        src_path: Path = Path(source_path)
        if not src_path.exists():
            raise NotFoundError(f"The source path({src_path}) does not exist.")

        dist_config = {
            "pp_degree": 1,
            "dp_degree": 1,
            "mp_degree": 1,
            "dp_mode": "allreduce",
            "parallelism_order": ["pp", "dp", "mp"],
        }

        attr = {}
        if attr_file_path is not None:
            try:
                with open(attr_file_path, "r", encoding="utf-8") as attr_f:
                    attr = yaml.safe_load(attr_f)
            except yaml.YAMLError as exc:
                raise InvalidConfigError(
                    f"The attribute YAML file has invalid format: {str(exc)}"
                ) from exc

        if attr:
            validate_checkpoint_attributes(attr)

        client = CheckpointClient()
        form_client = CheckpointFormClient()
        group_client = GroupProjectCheckpointClient()
        raw_ckpt_created = group_client.create_checkpoint(
            name=name,
            vendor=StorageType.FAI,
            region="",
            credential_id=None,
            iteration=iteration,
            storage_name="",
            files=[],
            dist_config=dist_config,
            attributes=attr,
        )
        ckpt_created = V1Checkpoint.model_validate(raw_ckpt_created)
        if not ckpt_created.forms:
            raise FriendliInternalError(
                f"No attached model forms to the checkpoint '{ckpt_created.id}'"
            )
        ckpt_form_id = ckpt_created.forms[0].id

        executor = ThreadPoolExecutor(max_workers=max_workers)
        adjuster = ChunksizeAdjuster()
        upload_manager = UploadManager(executor=executor, chunk_adjuster=adjuster)

        try:
            logger.info("Start uploading objects to create a checkpoint(%s)...", name)
            upload_local_src_paths = upload_manager.list_upload_objects(src_path)
            multipart_upload_local_src_paths = (
                upload_manager.list_multipart_upload_objects(src_path)
            )

            src_path = src_path if expand else src_path.parent
            upload_storage_dst_paths = [
                attach_storage_path_prefix(
                    path=p.relative_to(src_path),
                    iteration=iteration or 0,
                    mp_rank=0,
                    mp_degree=1,
                    pp_rank=0,
                    pp_degree=1,
                )
                for p in upload_local_src_paths
            ]
            multipart_upload_storage_dst_paths = [
                attach_storage_path_prefix(
                    path=p.relative_to(src_path),
                    iteration=iteration or 0,
                    mp_rank=0,
                    mp_degree=1,
                    pp_rank=0,
                    pp_degree=1,
                )
                for p in multipart_upload_local_src_paths
            ]
            upload_tasks = (
                [
                    UploadTask.model_validate(raw_url_info)
                    for raw_url_info in form_client.get_upload_urls(
                        obj_id=ckpt_form_id, storage_paths=upload_storage_dst_paths
                    )
                ]
                if len(upload_storage_dst_paths) > 0
                else []
            )
            multipart_upload_tasks = (
                [
                    MultipartUploadTask.model_validate(raw_url_info)
                    for raw_url_info in form_client.get_multipart_upload_urls(
                        obj_id=ckpt_form_id,
                        local_paths=multipart_upload_local_src_paths,
                        storage_paths=multipart_upload_storage_dst_paths,
                    )
                ]
                if len(multipart_upload_storage_dst_paths) > 0
                else []
            )

            for upload_task in upload_tasks:
                upload_manager.upload_file(
                    upload_task=upload_task, source_path=src_path
                )
            for multipart_upload_task in multipart_upload_tasks:
                upload_manager.multipart_upload_file(
                    upload_task=multipart_upload_task,
                    source_path=src_path,
                    complete_callback=functools.partial(
                        form_client.complete_multipart_upload, ckpt_form_id
                    ),
                    abort_callback=functools.partial(
                        form_client.abort_multipart_upload, ckpt_form_id
                    ),
                )

            files = [get_file_info(task.path, src_path) for task in upload_tasks]
            files.extend(
                [get_file_info(task.path, src_path) for task in multipart_upload_tasks]
            )
            form_client.update_checkpoint_files(ckpt_form_id=ckpt_form_id, files=files)

            # Activate the checkpoint.
            client.activate_checkpoint(ckpt_created.id)
        finally:
            logger.info("Checking the integrity of the checkpoint. Please wait...")
            raw_ckpt = client.get_checkpoint(ckpt_created.id)
            ckpt = V1Checkpoint.model_validate(raw_ckpt)
            if ckpt.status != CheckpointStatus.ACTIVE:
                logger.warn("File upload was unsuccessful. Please retry.")
                client.delete_checkpoint(ckpt.id)
            executor.shutdown(wait=True)

        logger.info(
            "Objects are uploaded and checkpoint(%s) is successfully created!", name
        )

        if ckpt.forms:
            for file in ckpt.forms[0].files:
                file.path = strip_storage_path_prefix(file.path)

        return ckpt

    def download(self, id: UUID, *, save_dir: Optional[str] = None) -> None:
        """Downloads a checkpoint to the local machine.

        Args:
            id (UUID): ID of checkpoint to donwload.
            save_dir (Optional[str], optional): Local direcotry path to save the checkpoint files. Defaults to None.

        Raises:
            NotFoundError: Raised when `save_dir` is not found.

        Examples:
            Basic usage:

            ```python
            from friendli import FriendliResource

            # Set up Friendli context.
            client = FriendliResource(
                token="YOUR_FRIENDLI_TOKEN",
                project_name="my-project",
            )

            client.checkpoint.download(
                id="YOUR_CHECKPOINT_ID",
                save_dir="local/save/dir",
            )
            ```

        """
        if save_dir is not None and not os.path.isdir(save_dir):
            raise NotFoundError(f"Directory {save_dir} is not found.")

        save_dir = save_dir or os.getcwd()

        client = CheckpointClient()
        form_client = CheckpointFormClient()
        ckpt_form_id = client.get_first_checkpoint_form(id)
        files = form_client.get_checkpoint_download_urls(ckpt_form_id)

        write_queue = DeferQueue()
        download_manager = DownloadManager(write_queue=write_queue)
        for i, file in enumerate(files):
            logger.info("Downloading files %d/%d...", i + 1, len(files))
            download_manager.download_file(
                url=file["download_url"],
                out=Path(save_dir) / strip_storage_path_prefix(file["path"]),
            )

    def restore(self, id: UUID) -> V1Checkpoint:
        """Restores a soft-deleted checkpoint.

        Args:
            id (UUID): ID of checkpoint to restore.

        Raises:
            NotFoundError: Raised when the checkpoint is not deleted.

        Returns:
            Dict[str, Any]: The restored checkpoint info.

        Examples:
            Basics usage:

            ```python
            from friendli import FriendliResource

            # Set up Friendli context.
            client = FriendliResource(
                token="YOUR_FRIENDLI_TOKEN",
                project_name="my-project",
            )

            checkpoint = client.checkpoint.restore(id="YOUR_CHECKPOINT_ID")
            ```

            :::info
            When a checkpoint is deleted, it becomes "soft-deleted", meaning it is
            recoverable within the 24-hour retention period. After the retention period,
            the checkpoint is "hard-deleted" and cannot be restored.
            :::

        """
        client = CheckpointClient()

        raw_ckpt = client.get_checkpoint(id)
        ckpt = V1Checkpoint.model_validate(raw_ckpt)
        if not ckpt.deleted:
            msg = f"Checkpoint({id}) is not deleted."
            raise NotFoundError(msg)

        raw_ckpt = client.restore_checkpoint(id)
        ckpt = V1Checkpoint.model_validate(raw_ckpt)
        return ckpt

    def import_from_catalog(
        self, id: UUID, *, name: str, method: CatalogImportMethod
    ) -> V1Checkpoint:
        """Tries out a public checkpoint in catalog.

        Args:
            id (UUID): ID of a catalog.
            name (str): The name of the checkpoint that will be created in the project.
            method (CatalogImportMethod): Import method.

        Returns:
            V1Checkpoint: The created checkpoint object by importing the public checkpoint in the catalog.

        """
        method = validate_enums(method, CatalogImportMethod)

        client = CatalogClient()
        raw_ckpt = client.try_out(catalog_id=id, name=name, method=method)
        ckpt = V1Checkpoint.model_validate(raw_ckpt)
        if ckpt.forms:
            for file in ckpt.forms[0].files:
                file.path = strip_storage_path_prefix(file.path)
        return ckpt
