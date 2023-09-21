# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint Converter."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, cast

import yaml

from periflow.enums import CheckpointDataType
from periflow.errors import (
    NotFoundError,
    NotSupportedCheckpointError,
    TokenizerNotFoundError,
)
from periflow.logging import logger
from periflow.modules.quantizer.schema import OneOfQuantConfig
from periflow.utils.validate import validate_convert_imports

validate_convert_imports()
# pylint: disable=import-outside-toplevel, wrong-import-position, wrong-import-order
import torch  # type: ignore[import]
import transformers  # type: ignore[import]

from periflow.modules.converter.maps import (  # pylint: disable=ungrouped-imports
    model_arch_converter_map,
)
from periflow.modules.converter.utils import (
    get_quantizer,
    get_tokenizer,
    save_tokenizer,
)
from periflow.modules.quantizer.utils import (
    get_encoded_dataset,
    get_quantized_state_dict,
)

# pylint: enable=import-outside-toplevel, wrong-import-position, wrong-import-order


def get_model_generation_config(
    model_name_or_path: str, cache_dir: Optional[str] = None
) -> Optional[transformers.GenerationConfig]:
    """Get HuggingFace model generation config."""
    try:
        generation_config = transformers.GenerationConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, trust_remote_code=True
        )
    except (OSError, TypeError):
        generation_config = None

    return generation_config


def get_model_pretrained_config(
    model_name_or_path: str, model_output_path: str, cache_dir: Optional[str] = None
) -> transformers.PretrainedConfig:
    """Get HuggingFace model configs."""
    try:
        config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, trust_remote_code=True
        )
    except OSError as exc:  # from transformers.AutoConfig.from_pretrained()
        config_dir = Path(model_name_or_path)
        model_output_dir = Path(model_output_path).parent
        if config_dir.exists() and model_output_dir.absolute() == config_dir.absolute():
            raise NotFoundError(
                f"'output_dir' ({model_output_dir.as_posix()}) and "
                f"'model_name_or_path' ({model_name_or_path}) are the same. "
                "In such a case, checkpoints should be prepared in 'output_dir'."
            ) from exc
        raise NotFoundError(str(exc)) from exc

    return config


def get_model_arch(config: transformers.PretrainedConfig) -> str:
    """Get HuggingFace model architecture from config."""
    model_arch_list = cast(
        List[str], cast(transformers.PretrainedConfig, config).architectures
    )
    if len(model_arch_list) == 0:
        raise NotSupportedCheckpointError(
            invalid_option=f"'architectures={model_arch_list}'",
            valid_options=list(model_arch_converter_map.keys()),
        )
    model_arch = model_arch_list[0]

    if model_arch not in model_arch_converter_map:
        raise NotSupportedCheckpointError(
            invalid_option=f"'architectures={model_arch}'",
            valid_options=list(model_arch_converter_map.keys()),
        )
    return model_arch


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

    if quantize and quant_config:
        quantizer = get_quantizer(model_arch, model_config, quant_config)
        quantizer.check_config()

    hf_factory, converter_factory = model_arch_converter_map[model_arch]
    converter = converter_factory(
        config=model_config,
        generation_config=generation_config,
        output_path=model_output_path,
        data_type=data_type,
        quantize=quantize,
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

        if quantize:
            logger.info(
                "Start SmoothQuant quantization for Hugging Face checkpoint...",
            )
            tokenizer = get_tokenizer(model_name_or_path, cache_dir=cache_dir)
            assert quant_config is not None
            dataset = get_encoded_dataset(quant_config, tokenizer)
            quantizer.pre_quantize(model, dataset, data_type)
            quant_result_iter = quantizer.quantize(model, dataset, data_type)
            state_dict = model.state_dict()
            state_dict.update(get_quantized_state_dict(quant_result_iter))
        else:
            state_dict = model.state_dict()
        converter.convert(state_dict=state_dict)
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
