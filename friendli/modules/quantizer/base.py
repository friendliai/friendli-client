# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Quantizer Base."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Dict, Iterator, List, Tuple, Type

import datasets  # type: ignore[import]
import huggingface_hub  # type: ignore[import]
import numpy as np
import torch

from friendli.enums import (
    QuantDatasetFormat,  # TODO: move this to friendli/modules/converter/enums.py
)
from friendli.errors import NotSupportedQuantConfigError
from friendli.modules.converter.base import DECODER_PREFIX, OneOfConverter
from friendli.modules.converter.interface import ModelConversionInterface
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.quantizer.schema.config import CommonQuantConfig
from friendli.modules.quantizer.schema.data import TFQuantInputs, TFQuantResults


class AbstractQuantHook(ABC):
    """Quantization Hook for a specific model architecture."""

    def __init__(self, quant_config: Dict[str, Any], converter: OneOfConverter):
        """Initialize the Quantization Hook.

        Args:
            quant_config: Quantization configuration.
            converter (OneOfConverter): Converter for a specific model architecture.
        """
        self.quant_config = quant_config
        self.converter = converter

    @abstractmethod
    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the transformer blocks."""

    @abstractmethod
    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the type of linear layer (etc. qkv, linear layer) in transformer block."""

    @abstractmethod
    def iter_tf_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer blocks."""

    @abstractmethod
    def get_quant_result(
        self,
        quant_inputs: TFQuantInputs,
        **kwargs: Any,
    ) -> TFQuantResults:
        """Returns the quantization result of the layer."""

    @property
    @abstractmethod
    def quantized_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for quantized layers."""

    @property
    @abstractmethod
    def modified_layers_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for modified layers."""

    @property
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block name."""
        return self.converter.decoder_layer_prefix

    @property
    def quantized_param_names(self) -> List[str]:
        """Return the parameter names of quantized layers."""
        param_names = []
        for i in range(self.converter.decoder_layer_num):
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            param_names.append(f"{converted_prefix}attn/c_attn/weight:0")
            param_names.append(f"{converted_prefix}attn/c_proj/weight:0")
            param_names.append(f"{converted_prefix}mlp/c_fc/weight:0")
            param_names.append(f"{converted_prefix}mlp/c_proj/weight:0")

        return param_names


class AbstractQuantizer(ABC):
    """Abstract Quantizer for a specific model architecture."""

    def __init__(
        self,
        hook: AbstractQuantHook,
        config: CommonQuantConfig,
        converter: OneOfConverter,
    ):
        """Initialize the Quantizer.

        Args:
            hook (AbstractQuantHook): Quantization Hook for a specific model architecture
            config (CommonQuantConfig): Quantization configuration.
            converter (OneOfConverter): Converter for a specific model architecture.

        """
        self.hook = hook
        self.quant_config = config
        self.converter = converter

    @abstractmethod
    def get_calib_dataset(
        self,
    ) -> datasets.Dataset:
        """Get calibration dataset."""

    @abstractmethod
    def pre_quantize(
        self,
        model: torch.nn.Module,
    ) -> None:
        """Pre-procedure that should be called before quantize() is called."""

    @abstractmethod
    def quantize(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        """Setting Quantizer from config and Quantize model."""


class CommonQuantizer(AbstractQuantizer, ModelConversionInterface):
    """Common Quantizer."""

    def check_config(self) -> None:
        """Check if the quantization config is valid."""
        self.converter.check_config()
        calibration_dataset_config = self.quant_config.calibration_dataset
        data_path_or_name = calibration_dataset_config.path_or_name
        percentile = self.quant_config.percentile
        if percentile <= 0 or percentile > 100:
            raise NotSupportedQuantConfigError(
                invalid_option=str(percentile),
                valid_options=["0 < percentile <= 100"],
            )
        if not os.path.exists(data_path_or_name):
            data_name = data_path_or_name.split(":")[0]
            if data_name not in (
                data.id for data in huggingface_hub.list_datasets(search=data_name)
            ):
                raise NotSupportedQuantConfigError(
                    invalid_option=data_name,
                    valid_options=["datasets on the huggingface hub", "local path"],
                )
        else:
            if calibration_dataset_config.format not in QuantDatasetFormat:
                raise NotSupportedQuantConfigError(
                    invalid_option=calibration_dataset_config.format,
                    valid_options=list(QuantDatasetFormat),
                )
        try:
            torch.device(self.quant_config.device)
        except ValueError as err:
            raise NotSupportedQuantConfigError(
                invalid_option=self.quant_config.device,
                valid_options=["cpu", "cuda"],
            ) from err

    def get_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Get List of the convert informations for the model."""
        convert_info_list = self.converter.get_convert_info_list()
        new_convert_info_list = []
        for convert_info in convert_info_list:
            if convert_info.converted_name in self.hook.quantized_param_names:
                continue
            new_convert_info_list.append(convert_info)

        return (
            new_convert_info_list
            + self.hook.quantized_convert_info_list
            + self.hook.modified_layers_convert_info_list
        )

    def get_attributes(self) -> Dict[str, Any]:
        """Return the attributes of the converted model."""
        return self.converter.get_attributes()

    def convert(
        self,
        model: torch.nn.Module,
        convert_info_list: List[ConvertInfo],
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Convert Huggingface Model to Friendli format(.h5).

        Args:
            model (torch.nn.Module): Huggingface model.
            state_dict (Dict[str, torch.Tensor]):
                Dictionary of mapping of tensor name to tensor
            convert_info_list (List[ConvertInfo]):
                Dictionary of mapping converted params name to conversion functions.

        """
        self.pre_quantize(model)
        model = self.quantize(model)
        yield from self.converter.convert(model, convert_info_list)
