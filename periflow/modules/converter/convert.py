# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint Converter."""

from __future__ import annotations

from typing import Optional

import yaml

from periflow.enums import CheckpointDataType
from periflow.errors import TokenizerNotFoundError
from periflow.logging import logger
from periflow.utils.validate import validate_convert_imports

validate_convert_imports()
# pylint: disable=import-outside-toplevel, wrong-import-position, wrong-import-order, ungrouped-imports
import torch  # type: ignore[import]

from periflow.modules.converter.maps import get_hf_converter_factory
from periflow.modules.converter.utils import (
    get_model_arch,
    get_model_generation_config,
    get_model_pretrained_config,
    save_tokenizer,
)
from periflow.modules.quantizer.maps import get_quantized_converter
from periflow.modules.quantizer.schema.config import OneOfQuantConfig

# pylint: enable=import-outside-toplevel, wrong-import-position, wrong-import-order, ungrouped-imports


def convert_checkpoint(  # pylint: disable=too-many-locals
    model_name_or_path: str,
    model_output_path: str,
    data_type: CheckpointDataType,
    *,
    tokenizer_output_dir: Optional[str] = None,
    attr_output_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dry_run: bool = False,
    quantize: bool = False,
    quant_config: Optional[OneOfQuantConfig] = None,
) -> None:
    """Convert HuggingFace model checkpoint to PeriFlow format.

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
    converter.check_config()

    if quantize:
        assert quant_config is not None
        converter = get_quantized_converter(  # type: ignore[assignment]
            model_arch, quant_config, converter
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

        convert_dict = converter.get_convert_dict()
        converter.convert(model, model_output_path, model.state_dict(), convert_dict)

        logger.info(
            "Hugging Face checkpoint(%s) is successfully converted to Periflow format!",
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
