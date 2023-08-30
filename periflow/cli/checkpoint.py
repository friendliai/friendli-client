# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=redefined-builtin, too-many-locals, too-many-arguments

"""PeriFlow Checkpoint CLI."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional
from uuid import UUID

import typer

from periflow.enums import (
    CheckpointCategory,
    CheckpointDataType,
    CheckpointStatus,
    StorageType,
)
from periflow.errors import (
    CheckpointConversionError,
    InvalidAttributesError,
    InvalidConfigError,
    InvalidPathError,
    NotFoundError,
    NotSupportedError,
)
from periflow.formatter import (
    JSONFormatter,
    PanelFormatter,
    TableFormatter,
    TreeFormatter,
)
from periflow.schema.resource.v1.checkpoint import V1Checkpoint
from periflow.sdk.resource.checkpoint import Checkpoint as CheckpointAPI
from periflow.utils.format import datetime_to_pretty_str, secho_error_and_exit

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)

table_formatter = TableFormatter(
    name="Checkpoints",
    fields=[
        "id",
        "name",
        "model_category",
        "iteration",
        "created_at",
    ],
    headers=[
        "ID",
        "Name",
        "Source",
        "Iteration",
        "Created At",
    ],
)
panel_formatter = PanelFormatter(
    name="Overview",
    fields=[
        "id",
        "name",
        "model_category",
        "forms[0].vendor",
        "forms[0].storage_name",
        "iteration",
        "forms[0].form_category",
        "created_at",
        "status",
    ],
    headers=[
        "ID",
        "Name",
        "Source",
        "Cloud",
        "Storage Name",
        "Iteration",
        "Format",
        "Created At",
        "Status",
    ],
)
json_formatter = JSONFormatter(name="Attributes")
tree_formatter = TreeFormatter(name="Files")


def get_translated_checkpoint_status(ckpt: V1Checkpoint) -> str:
    """Gets translated checkpoint status from the checkpoint info."""
    if ckpt.hard_deleted:
        hard_deleted_at = (
            datetime.strftime(ckpt.hard_deleted_at, "%Y-%m-%d %H:%M:%S")
            if ckpt.hard_deleted_at
            else "N/A"
        )
        typer.secho(
            f"This checkpoint was hard-deleted at {hard_deleted_at}. "
            "You cannot use this checkpoint.",
            fg=typer.colors.RED,
        )
        return "[bold red]Hard-Deleted"
    if ckpt.deleted:
        deleted_at = (
            datetime.strftime(ckpt.deleted_at, "%Y-%m-%d %H:%M:%S")
            if ckpt.deleted_at
            else "N/A"
        )
        typer.secho(
            f"This checkpoint was deleted at {deleted_at}. Please restore it with "
            f"'pf checkpoint restore {ckpt.id}' if you want use this.",
            fg=typer.colors.YELLOW,
        )
        return "[bold yellow]Soft-Deleted"
    if ckpt.status == CheckpointStatus.ACTIVE:
        return "[bold green]Active"
    return ckpt.status.value


@app.command()
def list(
    source: Optional[CheckpointCategory] = typer.Option(
        None,
        "--source",
        "-s",
        help="Source of checkpoints.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="The number of recent checkpoints to see.",
    ),
    deleted: bool = typer.Option(
        False, "--deleted", "-d", help="Shows deleted checkpoint."
    ),
):
    """Lists all checkpoints that belong to a user's organization."""
    checkpoints = CheckpointAPI.list(category=source, limit=limit, deleted=deleted)
    ckpt_dicts = []
    for ckpt in checkpoints:
        ckpt_dict = ckpt.model_dump()
        ckpt_dict["created_at"] = datetime_to_pretty_str(ckpt.created_at)
        ckpt_dicts.append(ckpt_dict)

    table_formatter.render(ckpt_dicts)


@app.command()
def view(
    checkpoint_id: UUID = typer.Argument(
        ..., help="ID of checkpoint to inspect in detail."
    )
):
    """Shows details of a checkpoint."""
    ckpt = CheckpointAPI.get(id=checkpoint_id)
    ckpt_dict = ckpt.model_dump()

    ckpt_dict["created_at"] = datetime_to_pretty_str(ckpt.created_at)
    ckpt_dict["status"] = get_translated_checkpoint_status(ckpt)

    panel_formatter.render([ckpt_dict])

    json_formatter.render(ckpt_dict["attributes"])
    tree_formatter.render(ckpt_dict["forms"][0]["files"])


@app.command()
def create(
    name: str = typer.Option(
        ..., "--name", "-n", help="Name of your checkpoint to create."
    ),
    credential_id: UUID = typer.Option(
        ...,
        "--credential-id",
        "-i",
        help="ID of crendential to access cloud storage.",
    ),
    cloud_storage: StorageType = typer.Option(
        ...,
        "--cloud-storage",
        "-c",
        help="The cloud storage vendor where the checkpoint is uploaded.",
    ),
    storage_name: str = typer.Option(
        ...,
        "--storage-name",
        "-s",
        help="The name of the cloud storage where the checkpoint is uploaded.",
    ),
    region: str = typer.Option(
        ...,
        "--region",
        "-r",
        help="The cloud storage region where the checkpoint is uploaded.",
    ),
    storage_path: Optional[str] = typer.Option(
        None,
        "--storage-path",
        "-p",
        help="File or directory path to cloud storage. Defaults to the root of the storage.",
    ),
    iteration: Optional[int] = typer.Option(
        None, "--iteration", help="The iteration number of the checkpoint."
    ),
    attr_file: Optional[str] = typer.Option(
        None,
        "--attr-file",
        "-f",
        help=(
            "Path to the file containing checkpoint attributes. The file should be in "
            "YAML format."
        ),
    ),
):
    """Creates a checkpoint by registering checkpoint files in user's cloud storage to PeriFlow.

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

    :::tip
    If you want to use tokenizer, `tokenizer.json` file should be in the same directory
    as checkpoint file(`*.h5`).
    :::

    """
    try:
        ckpt = CheckpointAPI.create(
            name=name,
            credential_id=credential_id,
            cloud_storage=cloud_storage,
            region=region,
            storage_name=storage_name,
            storage_path=storage_path,
            iteration=iteration,
            attr_file_path=attr_file,
        )
    except (InvalidConfigError, NotSupportedError, InvalidAttributesError) as exc:
        secho_error_and_exit(str(exc))

    ckpt_dict = ckpt.model_dump()
    ckpt_dict["created_at"] = datetime_to_pretty_str(ckpt.created_at)
    ckpt_dict["status"] = get_translated_checkpoint_status(ckpt)

    panel_formatter.render([ckpt_dict])
    json_formatter.render(ckpt_dict["attributes"])
    tree_formatter.render(ckpt_dict["forms"][0]["files"])


@app.command()
def delete(
    checkpoint_ids: List[UUID] = typer.Argument(
        ...,
        help=(
            "IDs of checkpoint to delete. "
            "When multiple IDs are provided in a space-separated string format, all "
            "corresponding checkpoints will be deleted."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Forcefully delete checkpoint without confirmation prompt.",
    ),
):
    """Deletes the existing checkpoint.

    :::info
    Deleting linked checkpoints created with `pf checkpoint create` simply unlinks the
    checkpoint from the PeriFlow system. That is, it does not physically delete the
    checkpoint objects from the source cloud storage (your cloud storage).

    Uploaded checkpoints are physically deleted.
    :::

    """
    targets_str = ""
    for checkpoint_id in checkpoint_ids:
        targets_str += str(checkpoint_id)
        targets_str += "\n"

    if not force:
        do_delete = typer.confirm(
            f"Following checkpoints will be deleted:\n\n{targets_str}\n"
            "Are your sure to delete these?"
        )
        if not do_delete:
            raise typer.Abort()

    for checkpoint_id in checkpoint_ids:
        CheckpointAPI.delete(id=checkpoint_id)

    typer.secho("Checkpoints are deleted successfully!", fg=typer.colors.BLUE)


@app.command()
def download(
    checkpoint_id: UUID = typer.Argument(..., help="ID of checkpoint to download."),
    save_directory: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path to the directory to save checkpoint files.",
    ),
):
    """Downloads checkpoint files to local storage.

    :::info
    You cannot download checkpoints linked from your cloud storage.
    :::

    """
    try:
        CheckpointAPI.download(id=checkpoint_id, save_dir=save_directory)
    except InvalidPathError as exc:
        secho_error_and_exit(str(exc))


@app.command()
def upload(
    name: str = typer.Option(
        ..., "--name", "-n", help="Name of the checkpoint to upload"
    ),
    source_path: str = typer.Option(
        ..., "--source-path", "-p", help="Path to source file or directory to upload"
    ),
    iteration: Optional[int] = typer.Option(
        None, "--iteration", help="The iteration number of the checkpoint."
    ),
    attr_file: Optional[str] = typer.Option(
        None,
        "--attr-file",
        "-f",
        help="Path to the file containing the checkpoint attributes. The file should "
        "be in YAML format.",
    ),
    max_workers: int = typer.Option(
        min(32, (os.cpu_count() or 1) + 4),  # default of ``ThreadPoolExecutor``
        "--max-workers",
        "-w",
        help="The number of threads to upload files.",
    ),
):
    """Creates a checkpoint by uploading local checkpoint files.

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

    :::tip
    If you want to use tokenizer, `tokenizer.json` file should be in the same directory
    as checkpoint file(`*.h5`).
    :::

    """
    try:
        ckpt = CheckpointAPI.upload(
            name=name,
            source_path=source_path,
            iteration=iteration,
            attr_file_path=attr_file,
            max_workers=max_workers,
        )
    except (InvalidPathError, InvalidConfigError, InvalidAttributesError) as exc:
        secho_error_and_exit(str(exc))

    ckpt_dict = ckpt.model_dump()
    ckpt_dict["created_at"] = datetime_to_pretty_str(ckpt.created_at)
    ckpt_dict["status"] = get_translated_checkpoint_status(ckpt)

    panel_formatter.render([ckpt_dict])
    json_formatter.render(ckpt_dict["attributes"])
    tree_formatter.render(ckpt_dict["forms"][0]["files"])


@app.command()
def restore(
    checkpoint_id: UUID = typer.Argument(..., help="ID of checkpoint to restore.")
):
    """Restores deleted checkpoint (available within 24 hours from the deletion).

    When a uploaded checkpoint is deleted, it is not physically deleted right away.
    Instead, it is soft-deleted and can be restored within 24 hours after the deletion.
    After the 24-hour retention period, it is hard-deleted and cannot be restored.

    :::caution
    When you delete a linked checkpoint, you cannot restore it. Instead, relink the
    original checkpoint using `pf checkpoint create`.
    :::

    """
    try:
        CheckpointAPI.restore(id=checkpoint_id)
    except NotFoundError as exc:
        secho_error_and_exit(str(exc))

    typer.secho(f"Checkpoint({checkpoint_id}) is successfully restored.")


@app.command()
def convert(
    model_name_or_path: str = typer.Option(
        ...,
        "--model-name-or-path",
        "-m",
        help="Hugging Face pretrained model name or path to the saved model checkpoint.",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help=(
            "Directory path to save the converted checkpoint and related configuration "
            "files. Three files will be created in the directory: `model.h5`, "
            "`tokenizer.json`, and `attr.yaml`. "
            "The `model.h5` is the converted checkpoint and can be renamed using "
            "the `--output-model-filename` option. "
            "The `tokenizer.json` is the PeriFlow-compatible tokenizer file, which should "
            "be uploaded along with the checkpoint file to tokenize the model input "
            "and output. "
            "The `attr.yaml` is the checkpoint attribute file, to be used when uploading "
            "the converted model to PeriFlow. You can designate the file name using "
            "the `--output-attr-filename` option."
        ),
    ),
    data_type: CheckpointDataType = typer.Option(
        ..., "--data-type", "-dt", help="The data type of converted checkpoint."
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help="Directory for downloading checkpoint."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only check conversion avaliability."
    ),
    output_model_file_name: str = typer.Option(
        "model.h5",
        "--output-model-filename",
        help="Name of the converted checkpoint file.",
    ),
    output_attr_file_name: str = typer.Option(
        "attr.yaml",
        "--output-attr-filename",
        help="Name of the checkpoint attribute file.",
    ),
):
    """Convert huggingface's model checkpoint to PeriFlow format.

    When a checkpoint is in the Hugging Face format, it cannot be directly served;
    rather, it requires conversion to the PeriFlow format for serving. The conversion
    process involves copying the original checkpoint and transforming it into a
    checkpoint in the PeriFlow format (*.h5).

    :::caution
    The `pf checkpoint convert` is available only when the package is installed with
    `pip install periflow-client[mllib]`.
    :::

    """
    try:
        from periflow.modules.converter.convert import (  # pylint: disable=import-outside-toplevel
            convert_checkpoint,
        )
    except ModuleNotFoundError as exc:
        secho_error_and_exit(str(exc))

    if not os.path.isdir(output_dir):
        if os.path.exists(output_dir):
            secho_error_and_exit(f"'{output_dir}' exists, but its not a directory.")
        os.mkdir(output_dir)

    model_output_path = os.path.join(output_dir, output_model_file_name)
    tokenizer_output_dir = output_dir
    attr_output_path = os.path.join(output_dir, output_attr_file_name)

    try:
        convert_checkpoint(
            model_name_or_path=model_name_or_path,
            model_output_path=model_output_path,
            data_type=data_type,
            tokenizer_output_dir=tokenizer_output_dir,
            attr_output_path=attr_output_path,
            cache_dir=cache_dir,
            dry_run=dry_run,
        )
    except (NotFoundError, CheckpointConversionError) as exc:
        secho_error_and_exit(str(exc))

    msg = (
        f"Checkpoint({model_name_or_path}) can be converted."
        if dry_run
        else f"Checkpoint({model_name_or_path}) is converted successfully."
    )
    typer.secho(msg)
