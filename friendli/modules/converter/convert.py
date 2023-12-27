# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Checkpoint Converter."""

from __future__ import annotations

from typing import Optional

import yaml
from peft import PeftModel  # type: ignore[import] # pylint: disable=import-error

from friendli.enums import CheckpointDataType, CheckpointFileType
from friendli.errors import TokenizerNotFoundError
from friendli.logging import logger
from friendli.modules.converter.saver import get_saver
from friendli.utils.validate import validate_convert_imports

validate_convert_imports()
# pylint: disable=import-outside-toplevel, wrong-import-position, wrong-import-order, ungrouped-imports
import torch  # type: ignore[import]
from accelerate import init_empty_weights  # type: ignore[import]

from friendli.modules.converter.maps import (
    get_adapter_converter_factory,
    get_hf_converter_factory,
)
from friendli.modules.converter.utils import (
    get_adapter_config,
    get_model_arch,
    get_model_generation_config,
    get_model_pretrained_config,
    save_tokenizer,
)
from friendli.modules.quantizer.maps import get_quantized_converter
from friendli.modules.quantizer.schema.config import OneOfQuantConfig

# pylint: enable=import-outside-toplevel, wrong-import-position, wrong-import-order, ungrouped-imports


def convert_checkpoint(  # pylint: disable=too-many-locals
    model_name_or_path: str,
    model_output_path: str,
    data_type: CheckpointDataType,
    output_ckpt_file_type: CheckpointFileType,
    *,
    tokenizer_output_dir: Optional[str] = None,
    attr_output_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dry_run: bool = False,
    quantize: bool = False,
    quant_config: Optional[OneOfQuantConfig] = None,
) -> None:
    """Convert HuggingFace model checkpoint to Friendli format.

    Args:
        model_name_or_path (str): Hugging Face model name or local path to the checkpoint.
        model_output_path (str): Path to the converted checkpoint to save.
        data_type (CheckpointDataType): Converted checkpoint data type.
        tokenizer_output_dir (Optional[str], optional): Directory path to save 'tokenizer.json'
            for the converted checkpoint. Defaults to None.
        attr_output_path (Optional[str], optional): Path to create the attribute YAML file for
            the converted checkpoint. Defaults to None.
        cache_dir (Optional[str], optional): Path for downloading checkpoint. Defaults to None.
        dry_run (bool, optional): Check only if checkpoint is convertable. Defaults to False.
        quantize (bool, optional): Enable quantization. Defaults to False.
        quant_config (Optional[OneOfQuantConfig], optional): Quantization configuration.
            Defaults to None.

    Raises:
        InValidconfigError: Raised when data_type is not supported.
        NotFoundError: Raised when `model_name_or_path` or `tokenizer_output_dir` is not found.
        NotSupportedCheckpointError: Raised when model architecture is not supported to convert.

    """
    model_config = get_model_pretrained_config(
        model_name_or_path, model_output_path, cache_dir
    )
    generation_config = get_model_generation_config(model_name_or_path, cache_dir)

    model_arch = get_model_arch(model_config)
    hf_factory, converter_factory = get_hf_converter_factory(model_arch)
    converter = converter_factory(
        config=model_config,
        generation_config=generation_config,
        data_type=data_type,
    )

    if quantize:
        assert quant_config is not None
        converter = get_quantized_converter(  # type: ignore[assignment]
            quant_config, converter
        )

    converter.check_config()

    if not dry_run:
        logger.info(
            "Start loading Hugging Face checkpoint(%s) for conversion...",
            model_name_or_path,
        )
        model = hf_factory.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            # `low_cpu_mem_usage` is for model loading faster and using ~1x model size CPU memory.
            # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained.example
        )

        logger.info(
            "Hugging Face checkpoint(%s) is successfully loaded!",
            model_name_or_path,
        )

        convert_info_list = converter.get_convert_info_list()
        with get_saver(output_ckpt_file_type, model_output_path) as saver:
            for name, w in converter.convert(model, convert_info_list):
                saver.save_tensor(name, w)

        logger.info(
            "Hugging Face checkpoint(%s) is successfully converted to Friendli format!",
            model_name_or_path,
        )

    if attr_output_path is not None:
        attr = converter.get_attributes()
        with open(attr_output_path, "w", encoding="utf-8") as file:
            yaml.dump(attr, file, sort_keys=False)

    if tokenizer_output_dir is not None:
        try:
            save_tokenizer(
                model_name_or_path=model_name_or_path,
                cache_dir=cache_dir,
                save_dir=tokenizer_output_dir,
            )
        except TokenizerNotFoundError as exc:
            logger.warn(str(exc))


def convert_adapter_checkpoint(  # pylint: disable=too-many-locals
    adapter_name_or_path: str,
    adapter_output_path: str,
    adapter_attr_output_path: str,
    data_type: CheckpointDataType,
    output_adapter_file_type: CheckpointFileType,
    cache_dir: Optional[str],
    dry_run: bool = False,
) -> None:
    """Convert HuggingFace model checkpoint to Friendli format."""
    adapter_config = get_adapter_config(adapter_name_or_path, cache_dir)
    model_config = get_model_pretrained_config(
        adapter_config.base_model_name_or_path, adapter_attr_output_path, cache_dir
    )
    model_arch = get_model_arch(model_config)
    hf_factory, converter_factory = get_hf_converter_factory(model_arch)
    converter = converter_factory(
        config=model_config,
        generation_config=None,
        data_type=data_type,
    )
    adapter_converter = get_adapter_converter_factory(model_arch)(
        converter, adapter_config
    )
    adapter_converter.check_config()

    if not dry_run:
        logger.info(
            "Start loading Hugging Face adapter checkpoint(%s's %s) for conversion...",
            adapter_config.base_model_name_or_path,
            adapter_name_or_path,
        )
        with init_empty_weights():
            model = hf_factory.from_pretrained(
                adapter_config.base_model_name_or_path,
                torch_dtype=torch.float32,
                cache_dir=cache_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        # inplace model update
        PeftModel.from_pretrained(
            model, adapter_name_or_path, cache_dir=cache_dir, torch_dtype=torch.float32
        )
        logger.info(
            "Hugging Face adapter checkpoint (%s) is successfully loaded!",
            adapter_name_or_path,
        )
        convert_dict = adapter_converter.get_convert_info_list()
        with get_saver(output_adapter_file_type, adapter_output_path) as saver:
            for name, w in adapter_converter.convert(model, convert_dict):
                saver.save_tensor(name, w)

        if adapter_attr_output_path is not None:
            attr = adapter_converter.get_attributes()
            with open(adapter_attr_output_path, "w", encoding="utf-8") as file:
                yaml.dump([attr], file, sort_keys=False)

        logger.info(
            "Hugging Face checkpoint (%s) is successfully converted to Friendli format!",
            adapter_name_or_path,
        )
