# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Base."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Tuple, Type

import datasets  # type: ignore[import]
import numpy as np
import torch

from periflow.enums import (
    QuantDatasetFormat,  # TODO: move this to periflow/modules/converter/enums.py
)
from periflow.errors import NotSupportedQuantConfigError
from periflow.modules.converter.base import OneOfConverter
from periflow.modules.converter.interface import ModelConversionInterface
from periflow.modules.quantizer.schema.config import CommonQuantConfig
from periflow.modules.quantizer.schema.data import TFQuantInputs, TFQuantResults


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
    def iter_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer blocks."""

    @abstractmethod
    def get_quant_result(
        self,
        quant_input: TFQuantInputs,
        **kwargs: Any,
    ) -> TFQuantResults:
        """Returns the quantization result of the layer."""

    @property
    @abstractmethod
    def quantized_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for quantized layers."""

    @property
    @abstractmethod
    def modified_layers_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for modified layers."""

    @property
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block name."""
        return self.converter.decoder_layer_prefix

    @property
    def quantized_param_names(self) -> List[str]:
        """Return the parameter names of quantized layers."""
        return [
            "attn/c_attn/weight:0",
            "attn/c_proj/weight:0",
            "mlp/c_fc/weight:0",
            "mlp/c_proj/weight:0",
        ]


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
    ) -> Iterator[TFQuantResults]:
        """Setting Quantizer from config and Quantize model."""

    @abstractmethod
    def get_quantized_state_dict(
        self,
        model: torch.nn.Module,
        quant_result_iter: Iterator[TFQuantResults],
    ) -> Dict[str, torch.Tensor]:
        """Get quantized state dict."""


class CommonQuantizer(AbstractQuantizer, ModelConversionInterface):
    """Common Quantizer."""

    def check_config(self) -> None:
        """Check if the quantization config is valid."""
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
            if data_name not in datasets.list_datasets():
                raise NotSupportedQuantConfigError(
                    invalid_option=data_name,
                    valid_options=[datasets.list_datasets(), "local path"],
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

    def get_convert_dict(
        self,
    ) -> Dict[str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]]:
        """Return the convert_dict for the model."""
        convert_dict = self.converter.get_convert_dict()
        for param_name in self.hook.quantized_param_names:
            del convert_dict["decoder"][param_name]

        convert_dict["decoder"].update(self.hook.quantized_convert_dict)
        convert_dict["decoder"].update(self.hook.modified_layers_convert_dict)
        return convert_dict

    def get_attributes(self) -> Dict[str, Any]:
        """Return the attributes of the converted model."""
        return self.converter.get_attributes()

    def convert(
        self,
        model: torch.nn.Module,
        output_path: str,
        state_dict: Dict[str, torch.Tensor],
        convert_dict: Dict[
            str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]
        ],
    ) -> None:
        """Convert Huggingface Model to PeriFlow format(.h5).

        Args:
            model (torch.nn.Module): Huggingface model.
            output_path (str): Path to save the converted model.
            state_dict (Dict[str, torch.Tensor]):
                Dictionary of mapping of tensor name to tensor
            convert_dict (Dict[Callable[[Dict[str, torch.Tensor], str], np.ndarray]]):
                Dictionary of mapping converted params name to conversion functions.

        """
        self.pre_quantize(model)
        quant_result_iter = self.quantize(model)
        state_dict = {
            **model.state_dict(),
            **self.get_quantized_state_dict(model, quant_result_iter),
        }

        self.converter.convert(model, output_path, state_dict, convert_dict)
