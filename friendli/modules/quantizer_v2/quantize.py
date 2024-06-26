# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Model Converter."""

from __future__ import annotations

from typing import Optional

from friendli.errors import TokenizerNotFoundError
from friendli.logging import logger
from friendli.modules.quantizer_v2.maps import get_hf_quantizer_factory
from friendli.modules.quantizer_v2.schema.config import OneOfQuantConfig
from friendli.modules.quantizer_v2.utils import (
    get_model_dtype,
    get_model_pretrained_config,
    save_tokenizer,
)


def quantize_checkpoint(
    model_name_or_path: str,
    output_dir: str,
    quant_config: OneOfQuantConfig,
    *,
    cache_dir: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Quantize HuggingFace model checkpoint to Friendli format.

    Args:
        model_name_or_path (str): Hugging Face model name or local path to the checkpoint.
        output_dir (str) : Directory path to save the converted checkpoint and the attribute YAML,
            and tokenizer configuration file.
        quant_config (OneOfQuantConfig): Quantization configuration.
        cache_dir (Optional[str], optional): Path for downloading checkpoint. Defaults to None.
        dry_run (bool, optional): Check only if checkpoint is convertable. Defaults to False.

    Raises:
        InValidconfigError: Raised when data_type is not supported.
        NotFoundError: Raised when `model_name_or_path` or `tokenizer_output_dir` is not found.
        NotSupportedCheckpointError: Raised when model architecture is not supported to quantize.
    """
    model_config = get_model_pretrained_config(
        model_name_or_path, output_dir, cache_dir
    )
    if quant_config.quant_scale_dtype is None:
        model_dtype = get_model_dtype(model_config.torch_dtype)
        quant_config.quant_scale_dtype = model_dtype
        logger.warn(
            "quant_scale_dtype is not set. Set to %s, same as hf model dtype.",
            model_dtype,
        )
    hf_factory, quantizer = get_hf_quantizer_factory(model_config, quant_config)
    dtype = model_config.torch_dtype
    quantizer.check_config()

    if not dry_run:
        logger.info(
            "Start loading Hugging Face checkpoint(%s) for conversion...",
            model_name_or_path,
        )
        model = hf_factory.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
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
        model = quantizer.quantize(model)
        model.config.update({"quantization_config": quantizer.get_quant_config()})
        model.save_pretrained(output_dir)
        try:
            save_tokenizer(
                model_name_or_path=model_name_or_path,
                cache_dir=cache_dir,
                save_dir=output_dir,
            )
        except TokenizerNotFoundError as exc:
            logger.warn(str(exc))
        logger.info(
            "Hugging Face checkpoint (%s) is successfully quantized to Friendli format!",
            model_name_or_path,
        )
