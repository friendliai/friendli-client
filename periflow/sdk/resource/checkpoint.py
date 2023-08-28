# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint SDK."""

# pylint: disable=line-too-long, arguments-differ, too-many-arguments, too-many-statements, too-many-locals, redefined-builtin

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import yaml

from periflow.client.checkpoint import CheckpointClient, CheckpointFormClient
from periflow.client.credential import CredentialClient
from periflow.client.group import GroupProjectCheckpointClient
from periflow.cloud.storage import build_storage_client
from periflow.enums import CheckpointCategory, CredType, StorageType
from periflow.errors import InvalidConfigError, NotFoundError, PeriFlowInternalError
from periflow.logging import logger
from periflow.schema.resource.v1.checkpoint import V1Checkpoint
from periflow.sdk.resource.base import ResourceAPI
from periflow.utils.fs import (
    FileSizeType,
    attach_storage_path_prefix,
    download_file,
    expand_paths,
    get_file_info,
    strip_storage_path_prefix,
)
from periflow.utils.maps import cred_type_map, cred_type_map_inv
from periflow.utils.validate import (
    validate_checkpoint_attributes,
    validate_cloud_storage_type,
    validate_enums,
    validate_storage_region,
)


class Checkpoint(ResourceAPI[V1Checkpoint, UUID]):
    """Checkpoint resource API."""

    @staticmethod
    def create(
        name: str,
        credential_id: UUID,
        cloud_storage: StorageType,
        region: str,
        storage_name: str,
        storage_path: Optional[str] = None,
        iteration: Optional[int] = None,
        attr_file_path: Optional[str] = None,
    ) -> V1Checkpoint:
        """Creates a checkpoint by linking the existing cloud storage (e.g., AWS S3, GCS, Azure Blob Storage) with PeriFlow.

        Args:
            name (str): The name of checkpoint to create.
            credential_id (UUID): Credential ID to access the cloud storage.
            cloud_storage (StorageType): Cloud storage type.
            region (str): Cloud region.
            storage_name (str): Storage name (e.g., AWS S3 bucket name).
            storage_path (Optional[str], optional): Path to the storage object (e.g., AWS S3 bucket key). Defaults to None.
            iteration (Optional[int], optional): The iteration of the checkpoint. Defaults to None.
            attr_file_path (Optional[str], optional): Path to the checkpoint attribute YAML file. Defaults to None.

        Returns:
            V1Checkpoint: Created checkpoint object.

        Raises:
            InvalidConfigError: Raised when checkpoint attribute file located at `attr_file_path` has invalid YAML format. Also raised when the credential with `credential_id` is not for the cloud provider of `cloud_storage`. Also raised when `region` is invalid.
            NotSupportedError: Raised when `cloud_storage` is not supported yet.
            InvalidAttributesError: Raised when the checkpoint attributes described in `attr_file_path` is in the invalid format.

        Examples:
            Basic usage:

            ```python
            import periflow as pf

            # Set up PeriFlow context.
            pf.init(
                api_key="YOUR_PERILFOW_API_KEY",
                project_name="my-project",
            )

            # Create a checkpoint by linking an existing S3 bucket.
            checkpoint = pf.Checkpoint.create(
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

    @staticmethod
    def list(
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
            import periflow as pf

            # Set up PeriFlow context.
            pf.init(
                api_key="YOUR_PERILFOW_API_KEY",
                project_name="my-project",
            )

            checkpoints = pf.Checkpoint.list(limit=100)
            ```

            To get the deleted checkpoints created by users:

            ```python
            checkpoints = pf.Checkpoint.list(
                category=CheckpointCategory.USER_PROVIDED, deleted=True
            )
            ```

        """
        client = GroupProjectCheckpointClient()
        checkpoints = [
            V1Checkpoint.model_validate(raw_ckpt)
            for raw_ckpt in client.list_checkpoints(
                category, limit=limit, deleted=deleted
            )
        ]
        return checkpoints

    @staticmethod
    def get(id: UUID, *args, **kwargs) -> V1Checkpoint:
        """Gets a specific checkpoint.

        Args:
            id (UUID): ID of checkpoint to retrieve.

        Returns:
            V1Checkpoint: The retrieved checkpoint object.

        Examples:
            Basic usage:

            ```python
            import periflow as pf

            # Set up PeriFlow context.
            pf.init(
                api_key="YOUR_PERILFOW_API_KEY",
                project_name="my-project",
            )

            checkpoint = pf.Checkpoint.get(id="YOUR_CHECKPOINT_ID")
            ```

        """
        client = CheckpointClient()
        raw_ckpt = client.get_checkpoint(id)
        ckpt = V1Checkpoint.model_validate(raw_ckpt)
        if ckpt.forms:
            for file in ckpt.forms[0].files:
                file.path = strip_storage_path_prefix(file.path)
        return ckpt

    @staticmethod
    def delete(id: UUID) -> None:
        """Deletes a checkpoint.

        Args:
            id (UUID): ID of checkpoint to delete.

        Examples:
            Basic usage:

            ```python
            import periflow as pf

            # Set up PeriFlow context.
            pf.init(
                api_key="YOUR_PERILFOW_API_KEY",
                project_name="my-project",
            )

            checkpoint = pf.Checkpoint.delete(id="YOUR_CHECKPOINT_ID")
            ```

        """
        client = CheckpointClient()
        client.delete_checkpoint(id)

    @staticmethod
    def upload(
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

        Returns:
            V1Checkpoint: Created checkpoint object.

        Raises:
            NotFoundError: Raised when `source_path` does not exist.
            InvalidConfigError: Raised when the attribute file located at `attr_file_path` has invalid YAML format.
            InvalidAttributesError: Raised when the checkpoint attributes described in `attr_file_path` is in the invalid format.

        Examples:
            Basic usage:

            ```python
            import periflow as pf

            # Set up PeriFlow context.
            pf.init(
                api_key="YOUR_PERILFOW_API_KEY",
                project_name="my-project",
            )

            # Create a checkpoint by loading files located at "local/path/to/ckpt" directory.
            checkpoint = pf.Checkpoint.upload(
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
        raw_ckpt = group_client.create_checkpoint(
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
        ckpt = V1Checkpoint.model_validate(raw_ckpt)
        if not ckpt.forms:
            raise PeriFlowInternalError(
                f"No attached model forms to the checkpoint '{ckpt.id}'"
            )
        ckpt_form_id = ckpt.forms[0].id

        try:
            logger.info("Start uploading objects to create a checkpoint(%s)...", name)
            spu_local_paths = expand_paths(src_path, FileSizeType.SMALL)
            mpu_local_paths = expand_paths(src_path, FileSizeType.LARGE)
            src_path = src_path if expand else src_path.parent
            # TODO: Need to support distributed checkpoints for model parallelism.
            spu_storage_paths = [
                attach_storage_path_prefix(
                    path=str(Path(p).relative_to(src_path)),
                    iteration=iteration or 0,
                    mp_rank=0,
                    mp_degree=1,
                    pp_rank=0,
                    pp_degree=1,
                )
                for p in spu_local_paths
            ]
            mpu_storage_paths = [
                attach_storage_path_prefix(
                    path=str(Path(p).relative_to(src_path)),
                    iteration=iteration or 0,
                    mp_rank=0,
                    mp_degree=1,
                    pp_rank=0,
                    pp_degree=1,
                )
                for p in mpu_local_paths
            ]
            spu_url_dicts = (
                form_client.get_spu_urls(
                    obj_id=ckpt_form_id, storage_paths=spu_storage_paths
                )
                if len(spu_storage_paths) > 0
                else []
            )
            mpu_url_dicts = (
                form_client.get_mpu_urls(
                    obj_id=ckpt_form_id,
                    local_paths=mpu_local_paths,
                    storage_paths=mpu_storage_paths,
                )
                if len(mpu_storage_paths) > 0
                else []
            )

            form_client.upload_files(
                obj_id=ckpt_form_id,
                spu_url_dicts=spu_url_dicts,
                mpu_url_dicts=mpu_url_dicts,
                source_path=src_path,
                max_workers=max_workers,
            )

            files = [
                get_file_info(url_info["path"], src_path) for url_info in spu_url_dicts
            ]
            files.extend(
                [
                    get_file_info(url_info["path"], src_path)
                    for url_info in mpu_url_dicts
                ]
            )
            form_client.update_checkpoint_files(ckpt_form_id=ckpt_form_id, files=files)
        except Exception as exc:
            client.delete_checkpoint(checkpoint_id=ckpt.id)
            raise exc

        # Activate the checkpoint.
        raw_ckpt = client.activate_checkpoint(ckpt.id)
        ckpt = V1Checkpoint.model_validate(raw_ckpt)
        if ckpt.forms:
            for file in ckpt.forms[0].files:
                file.path = strip_storage_path_prefix(file.path)

        logger.info(
            "Objects are uploaded and checkpoint(%s) is successfully created!", name
        )
        return ckpt

    @staticmethod
    def download(id: UUID, save_dir: Optional[str] = None) -> None:
        """Downloads a checkpoint to the local machine.

        Args:
            id (UUID): ID of checkpoint to donwload.
            save_dir (Optional[str], optional): Local direcotry path to save the checkpoint files. Defaults to None.

        Raises:
            NotFoundError: Raised when `save_dir` is not found.

        Examples:
            Basic usage:

            ```python
            import periflow as pf

            # Set up PeriFlow context.
            pf.init(
                api_key="YOUR_PERILFOW_API_KEY",
                project_name="my-project",
            )

            pf.Checkpoint.download(
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

        for i, file in enumerate(files):
            logger.info("Downloading files {%d}/{%d}...", i + 1, len(files))
            download_file(
                url=file["download_url"],
                out=os.path.join(save_dir, strip_storage_path_prefix(file["path"])),
            )

    @staticmethod
    def restore(id: UUID) -> V1Checkpoint:
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
            import periflow as pf

            # Set up PeriFlow context.
            pf.init(
                api_key="YOUR_PERILFOW_API_KEY",
                project_name="my-project",
            )

            pf.Checkpoint.download(
                id="YOUR_CHECKPOINT_ID",
                save_dir="local/save/dir",
            )
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
