# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=redefined-builtin, too-many-locals, too-many-arguments, line-too-long

"""Friendli Checkpoint CLI."""

from __future__ import annotations

import os
import random
import string
from typing import List, Optional, cast
from uuid import UUID

import typer
import yaml

from friendli.enums import (
    CatalogImportMethod,
    CheckpointCategory,
    CheckpointDataType,
    CheckpointFileType,
    StorageType,
)
from friendli.errors import (
    CheckpointConversionError,
    InvalidAttributesError,
    InvalidConfigError,
    InvalidPathError,
    NotFoundError,
    NotSupportedError,
)
from friendli.formatter import (
    JSONFormatter,
    PanelFormatter,
    TableFormatter,
    TreeFormatter,
)
from friendli.sdk.resource.catalog import Catalog
from friendli.sdk.resource.checkpoint import Checkpoint
from friendli.utils.decorator import check_api
from friendli.utils.format import (
    datetime_to_pretty_str,
    get_translated_checkpoint_status,
    secho_error_and_exit,
)

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


# @app.command()
@check_api
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
    client = Checkpoint()
    checkpoints = client.list(category=source, limit=limit, deleted=deleted)
    ckpt_dicts = []
    for ckpt in checkpoints:
        ckpt_dict = ckpt.model_dump()
        ckpt_dict["created_at"] = datetime_to_pretty_str(ckpt.created_at)
        ckpt_dicts.append(ckpt_dict)

    table_formatter.render(ckpt_dicts)


# @app.command()
@check_api
def view(
    checkpoint_id: UUID = typer.Argument(
        ..., help="ID of checkpoint to inspect in detail."
    )
):
    """Shows details of a checkpoint."""
    client = Checkpoint()
    ckpt = client.get(id=checkpoint_id)
    ckpt_dict = ckpt.model_dump()

    ckpt_dict["created_at"] = datetime_to_pretty_str(ckpt.created_at)
    ckpt_dict["status"] = get_translated_checkpoint_status(ckpt)

    panel_formatter.render([ckpt_dict])
    json_formatter.render(ckpt_dict["attributes"])
    tree_formatter.render(ckpt_dict["forms"][0]["files"])


# @app.command()
@check_api
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
    """Creates a checkpoint by registering checkpoint files in user's cloud storage to Friendli.

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
    client = Checkpoint()
    try:
        ckpt = client.create(
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


# @app.command()
@check_api
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
    Deleting linked checkpoints created with `friendli checkpoint create` simply unlinks the
    checkpoint from the Friendli system. That is, it does not physically delete the
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

    client = Checkpoint()
    for checkpoint_id in checkpoint_ids:
        client.delete(id=checkpoint_id)

    typer.secho("Checkpoints are deleted successfully!", fg=typer.colors.BLUE)


# @app.command()
@check_api
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
    client = Checkpoint()
    try:
        client.download(id=checkpoint_id, save_dir=save_directory)
    except InvalidPathError as exc:
        secho_error_and_exit(str(exc))


# @app.command()
@check_api
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
    client = Checkpoint()
    try:
        ckpt = client.upload(
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


# @app.command()
@check_api
def restore(
    checkpoint_id: UUID = typer.Argument(..., help="ID of checkpoint to restore.")
):
    """Restores deleted checkpoint (available within 24 hours from the deletion).

    When a uploaded checkpoint is deleted, it is not physically deleted right away.
    Instead, it is soft-deleted and can be restored within 24 hours after the deletion.
    After the 24-hour retention period, it is hard-deleted and cannot be restored.

    :::caution
    When you delete a linked checkpoint, you cannot restore it. Instead, relink the
    original checkpoint using `friendli checkpoint create`.
    :::

    """
    client = Checkpoint()
    try:
        client.restore(id=checkpoint_id)
    except NotFoundError as exc:
        secho_error_and_exit(str(exc))

    typer.secho(f"Checkpoint({checkpoint_id}) is successfully restored.")


# @app.command("import")
@check_api
def import_from_catalog(
    catalog_name: str = typer.Argument(
        ...,
        help="The name of public checkpoint to try out.",
    ),
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="The name of the checkpoint that will be created in the project.",
    ),
    method: CatalogImportMethod = typer.Option(
        CatalogImportMethod.COPY.value,
        "--method",
        "-m",
        help=(
            "The method to import the public checkpoint. When the method is 'copy', "
            "the file objects will be copied to a separate storage so that the "
            "checkpoint remains available even if the source public checkpoint is "
            "removed from the catalog. Conversely, when the 'ref' method is selected, "
            "the file objects are referenced from the storage of the source public "
            "checkpoint. In this case, the imported checkpoint becomes unavailable if "
            "the source in the catalog is deleted."
        ),
    ),
):
    """Create a checkpoint by importing a public checkpoint in the catalog."""
    catalog_client = Catalog()
    catalogs = catalog_client.list(name=catalog_name)
    catalog = None
    for cat in catalogs:
        if cat.name == catalog_name:
            catalog = cat
            break

    if catalog is None:
        msg = (
            f"Public checkpoint with name '{catalog_name}' is not found in the catalog."
        )
        if len(catalogs) > 0:
            msg += f" Did you mean '{catalogs[0].name}'?"
        secho_error_and_exit(msg)

    if name is None:
        hash = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
        name = f"{catalog.name}-{hash}"

    ckpt_client = Checkpoint()
    ckpt = ckpt_client.import_from_catalog(id=catalog.id, name=name, method=method)
    ckpt_dict = ckpt.model_dump()

    ckpt_dict["created_at"] = datetime_to_pretty_str(ckpt.created_at)
    status = get_translated_checkpoint_status(ckpt)
    ckpt_dict["status"] = status

    panel_formatter.render([ckpt_dict])
    json_formatter.render(ckpt_dict["attributes"])
    tree_formatter.render(ckpt_dict["forms"][0]["files"])


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
            "The `model.h5` or `model.safetensors` is the converted checkpoint and can be renamed using "
            "the `--output-model-filename` option. "
            "The `tokenizer.json` is the Friendli-compatible tokenizer file, which should "
            "be uploaded along with the checkpoint file to tokenize the model input "
            "and output. "
            "The `attr.yaml` is the checkpoint attribute file, to be used when uploading "
            "the converted model to Friendli. You can designate the file name using "
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
        None,
        "--output-model-filename",
        help="Name of the converted checkpoint file."
        "The default file name is `model.h5` when `--output-ckpt-file-type` is `hdf5` or `model.safetensors` when `--output-ckpt-file-type` is `safetensors`.",
    ),
    output_ckpt_file_type: CheckpointFileType = typer.Option(
        CheckpointFileType.HDF5,
        "--output-ckpt-file-type",
        help="File format of the converted checkpoint file.",
    ),
    output_attr_file_name: str = typer.Option(
        "attr.yaml",
        "--output-attr-filename",
        help="Name of the checkpoint attribute file.",
    ),
    quantize: bool = typer.Option(
        False,
        "--quantize",
        help="Quantize the model before conversion",
    ),
    quant_config_file: Optional[typer.FileText] = typer.Option(
        None,
        "--quant-config-file",
        help="Path to the quantization configuration file.",
    ),
):
    """Convert huggingface's model checkpoint to Friendli format.

    When a checkpoint is in the Hugging Face format, it cannot be directly served.
    It requires conversion to the Friendli format for serving. The conversion
    process involves copying the original checkpoint and transforming it into a
    checkpoint in the Friendli format (*.h5).

    :::caution
    The `friendli checkpoint convert` is available only when the package is installed with
    `pip install "friendli-client[mllib]"`.
    :::

    ### Apply quantization

    If you want to quantize the model along with the conversion, `--quantize` option
    should be provided. You can customize the quantization configuration by describing
    it in a YAML file and providing the path to the file to `--quant-config-file`
    option. When `--quantize` option is used without providing `--quant-config-file`,
    the following configuration is used by default.

    ```yaml
    # Default quantization configuration
    mode: awq
    device: cuda:0
    seed: 42
    offload: true
    calibration_dataset:
        path_or_name: lambada
        format: json
        split: validation
        lookup_column_name: text
        num_samples: 128
        max_length: 512
    awq_args:
        quant_bit: 4
        quant_group_size: 64
    ```

    - **`mode`**: Quantization scheme to apply. Defaults to "awq".
    - **`device`**: Device to run the quantization process. Defaults to "cuda:0".
    - **`seed`**: Random seed. Defaults to 42.
    - **`offload`**: When enabled, this option significantly reduces GPU memory usage by offloading model layers onto CPU RAM. Defaults to true.
    - **`calibration_dataset`**
        - **`path_or_name`**: Path or name of the dataset. Datasets from either the Hugging Face Datasets Hub or local file system can be used. Defaults to "lambada".
        - **`format`**: Format of datasets. Defaults to "json".
        - **`split`**: Which split of the data to load. Defaults to "validation".
        - **`lookup_column_name`**: The name of a column in the dataset to be used as calibration inputs. Defaults to "text".
        - **`num_samples`**: The number of dataset samples to use for calibration. Note that the dataset will be shuffled before sampling. Defaults to 512.
        - **`max_length`**: The maximum length of a calibration input sequence. Defauts to 512.
    - **`awq_args`** (Fill in this field only for "awq" mode)
        - **`quant_bit`** : Bit width of integers to represent weights. Possible values are `4` or `8`. Defaults to 4.
        - **`quant_group_size`**: Group size of quantized matrices. 64 is the only supported value at this time. Defaults to 64.

    :::tip
    If you encounter OOM issues when running with AWQ, try enabling the `offload` option.
    :::

    :::tip
    If you set `percentile` in quant-config-file into 100,
    the quantization range will be determined by the maximum absolute values of the activation tensors.
    :::

    :::info
    Currently, [AWQ](https://arxiv.org/abs/2306.00978) is the only supported quantization scheme.
    :::

    :::info
    AWQ is supported only for models with architecture listed as follows:

    - `GPTNeoXForCausalLM`
    - `GPTJForCausalLM`
    - `LlamaForCausalLM`
    - `MPTForCausalLM`
    :::

    """
    try:
        from friendli.modules.converter.convert import (  # pylint: disable=import-outside-toplevel
            convert_checkpoint,
        )
        from friendli.modules.quantizer.schema.config import (  # pylint: disable=import-outside-toplevel
            AWQConfig,
            OneOfQuantConfig,
            QuantConfig,
        )
    except ModuleNotFoundError as exc:
        secho_error_and_exit(str(exc))

    if not os.path.isdir(output_dir):
        if os.path.exists(output_dir):
            secho_error_and_exit(f"'{output_dir}' exists, but it is not a directory.")
        os.mkdir(output_dir)

    quant_config: Optional[OneOfQuantConfig] = None
    if quantize:
        if quant_config_file:
            try:
                quant_config_dict = cast(dict, yaml.safe_load(quant_config_file.read()))
            except yaml.YAMLError as err:
                secho_error_and_exit(f"Failed to load the quant config file: {err}")
            quant_config = QuantConfig.model_validate(
                {"config": quant_config_dict}
            ).config
        else:
            quant_config = AWQConfig()

    default_names = {
        CheckpointFileType.HDF5: "model.h5",
        CheckpointFileType.SAFETENSORS: "model.safetensors",
    }
    output_model_file_name = (
        output_model_file_name or default_names[output_ckpt_file_type]
    )

    model_output_path = os.path.join(output_dir, output_model_file_name)
    tokenizer_output_dir = output_dir
    attr_output_path = os.path.join(output_dir, output_attr_file_name)

    try:
        convert_checkpoint(
            model_name_or_path=model_name_or_path,
            model_output_path=model_output_path,
            data_type=data_type,
            output_ckpt_file_type=output_ckpt_file_type,
            tokenizer_output_dir=tokenizer_output_dir,
            attr_output_path=attr_output_path,
            cache_dir=cache_dir,
            dry_run=dry_run,
            quantize=quantize,
            quant_config=quant_config,
        )
    except (NotFoundError, CheckpointConversionError, InvalidConfigError) as exc:
        secho_error_and_exit(str(exc))

    msg = (
        f"Checkpoint({model_name_or_path}) can be converted."
        if dry_run
        else f"Checkpoint({model_name_or_path}) has been converted successfully."
    )
    typer.secho(msg)


@app.command()
def convert_adapter(
    adapter_name_or_path: str = typer.Option(
        ...,
        "--adapter-name-or-path",
        "-a",
        help="Hugging Face pretrained adapter name or path to the saved adapter checkpoint.",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help=(
            "Directory path to save the converted adapter checkpoint and related configuration "
            "files. Two files will be created in the directory: `adapter.h5`, "
            "and `attr.yaml`. "
            "The `adapter.h5` is the converted checkpoint and can be renamed using "
            "the `--output-adapter-filename` option. "
            "The `attr.yaml` is the adapter checkpoint attribute file, to be used when uploading "
            "the converted model to Friendli. You can designate the file name using "
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
    output_adapter_filename: str = typer.Option(
        "adapter.h5",
        "--output-adapter-filename",
        help="Name of the converted adapter checkpoint file.",
    ),
    output_attr_filename: str = typer.Option(
        "adapter_attr.yaml",
        "--output-attr-filename",
        help="Name of the adapter checkpoint attribute file.",
    ),
) -> None:
    """Convert huggingface's adapter checkpoint to Friendli format.

    When an adapter checkpoint is in the Hugging Face PEFT format, it cannot
    be directly served in Friendli. It requires conversion to the Friendli format.
    The conversion process involves copying the original adapter checkpoint and
    transforming it into a checkpoint in the Friendli format (*.h5).

    This function does not include the `friendli checkpoint convert` command. i.e.
    `friendli checkpoint convert-adapter` only converts adapter's parameters, not backbone's.

    :::caution
    The `friendli checkpoint convert-adapter` is available only when the package is installed with
    `pip install "friendli-client[mllib]"`.
    :::

    """
    try:
        from friendli.modules.converter.convert import (  # pylint: disable=import-outside-toplevel
            convert_adapter_checkpoint,
        )
    except ModuleNotFoundError as exc:
        secho_error_and_exit(str(exc))

    if not os.path.isdir(output_dir):
        if os.path.exists(output_dir):
            secho_error_and_exit(f"'{output_dir}' exists, but it is not a directory.")
        os.mkdir(output_dir)

    # Engine cannot load a Safetensors Lora ckpt yet.
    output_adapter_file_type = CheckpointFileType.HDF5
    default_names = {
        CheckpointFileType.HDF5: "adapter.h5",
        CheckpointFileType.SAFETENSORS: "adapter.safetensors",
    }
    output_adapter_filename = (
        output_adapter_filename or default_names[output_adapter_file_type]
    )

    adapter_output_path = os.path.join(output_dir, output_adapter_filename)
    attr_output_path = os.path.join(output_dir, output_attr_filename)

    try:
        convert_adapter_checkpoint(
            adapter_name_or_path=adapter_name_or_path,
            adapter_output_path=adapter_output_path,
            adapter_attr_output_path=attr_output_path,
            data_type=data_type,
            output_adapter_file_type=output_adapter_file_type,
            cache_dir=cache_dir,
            dry_run=dry_run,
        )
    except (NotFoundError, CheckpointConversionError, InvalidConfigError) as exc:
        secho_error_and_exit(str(exc))
