# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=redefined-builtin, too-many-locals, too-many-arguments, line-too-long

"""Friendli Checkpoint CLI."""

from __future__ import annotations

import os
from typing import Optional, cast

import typer
import yaml

from friendli.enums import CheckpointDataType, CheckpointFileType
from friendli.errors import CheckpointConversionError, InvalidConfigError, NotFoundError
from friendli.formatter import (
    JSONFormatter,
    PanelFormatter,
    TableFormatter,
    TreeFormatter,
)
from friendli.utils.compat import model_parse
from friendli.utils.format import secho_error_and_exit

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
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
            quant_config = model_parse(
                QuantConfig, {"config": quant_config_dict}
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
