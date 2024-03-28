# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Quantizer Base."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple, Type, Union, cast

import datasets  # type: ignore[import]
import huggingface_hub  # type: ignore[import]
import numpy as np
import torch
from torch.nn.modules import Module
from tqdm import tqdm

from friendli.enums import (
    QuantDatasetFormat,  # TODO: move this to friendli/modules/converter/enums.py
)
from friendli.enums import ModelDataType
from friendli.errors import NotSupportedQuantConfigError
from friendli.logging import logger
from friendli.modules.converter.base import DECODER_PREFIX, OneOfConverter
from friendli.modules.converter.interface import ModelConversionInterface
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.converter.utils import get_tokenizer, get_torch_data_type
from friendli.modules.quantizer.layers import WeightActQuantizedLinearLayer
from friendli.modules.quantizer.schema.config import OneOfQuantConfig
from friendli.modules.quantizer.schema.data import (
    HFTFQuantInputs,
    ModuleName,
    TFQuantInputs,
    TFQuantResults,
    WeightActQuantResult,
)
from friendli.modules.quantizer.utils import (
    collect_stats,
    offload_module_sequence,
    safe_load_datasets,
    send_model_to_device,
)


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
    def iter_tf_quant_inputs(
        self, model: torch.nn.Module
    ) -> Union[Iterator[TFQuantInputs], Iterator[HFTFQuantInputs]]:
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
        config: OneOfQuantConfig,
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

    @contextmanager
    def _try_offload_model(self, model: torch.nn.Module):
        if not self.quant_config.offload:
            logger.info("Offloading not enabled. Skipping.")
            model.to(self.quant_config.device)
            yield
        else:
            logger.info("Offloading enabled.")
            tf_blocks = self.hook.get_tf_blocks(model)
            send_model_to_device(model, self.quant_config.device, exclude=tf_blocks)
            with offload_module_sequence(tf_blocks, self.quant_config.device):
                yield

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


class FP8QuantHook(AbstractQuantHook):
    """Quantization Hook for FP8Quantizer."""

    def get_quant_result(
        self, quant_inputs: TFQuantInputs, **kwargs: Any
    ) -> TFQuantResults:
        """Returns the quantization result of the layer."""
        raise NotImplementedError

    @property
    def quantized_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for quantized layers."""
        raise NotImplementedError

    @property
    def modified_layers_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for modified layers."""
        raise NotImplementedError


class FP8Quantizer(CommonQuantizer):
    """Common Quantizer for huggingface format.

    This quantizer supports per-tensor weight-activation quantization by
    using calibration dataset. It adds quantization scale, and quantized
    parameter to the checkpoint, while preserves parameter shape, and name
    in huggingface checkpoint.
    """

    def get_calib_dataset(self) -> datasets.Dataset:
        """Get calibration dataset."""
        data_cfg = self.quant_config.calibration_dataset
        tokenizer = get_tokenizer(self.converter.config.name_or_path)
        dataset = safe_load_datasets(data_cfg)

        dataset = (
            dataset.shuffle(self.quant_config.seed)
            .select(range(data_cfg.num_samples))
            .select_columns([data_cfg.lookup_column_name])
        )

        encoded_dataset = tokenizer(
            dataset["article"][:512],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=data_cfg.max_length,
        )
        return encoded_dataset["input_ids"]

    def get_convert_info_list(self) -> List[ConvertInfo]:
        """Not used in FP8Quantizer."""
        return []

    def pre_quantize(self, model: Module) -> None:
        """Not used in FP8Quantizer."""
        return None

    def _get_weight_act_quantize_results(
        self,
        model: torch.nn.Module,
        names: List[ModuleName],
        max_input_stats: Dict[ModuleName, torch.Tensor],
    ) -> List[WeightActQuantResult]:
        """Get the quantization scales and quantized_weight for a specific layer."""
        assert (
            self.quant_config.quant_dtype == ModelDataType.FP8_E4M3
        ), "currently support fp8_e4m3"
        max_val = 448.0
        min_val = -448.0

        input_max = torch.concat([max_input_stats[name] for name in names])
        target_weights = [model.get_submodule(name).weight for name in names]
        target_weight = torch.concat(target_weights)

        act_scale = float(input_max.detach().abs().max().item()) / float(max_val)
        weight_scale = float(target_weight.detach().abs().max().item()) / float(max_val)

        q_weights = [
            ((weight.detach().float() / weight_scale).clip(min_val, max_val).to("cpu"))
            for weight in target_weights
        ]
        return [
            WeightActQuantResult(
                name,
                quant_dtype=self.quant_config.quant_dtype,
                act_scale=torch.tensor(act_scale, dtype=torch.float32),
                weight_scale=torch.tensor(weight_scale, dtype=torch.float32),
                q_weight=q_weight,
                q_group_size=-1,
                zero_point=torch.tensor(0.0),
            )
            for name, q_weight in zip(names, q_weights)
        ]

    @torch.no_grad()
    def quantize(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        """Quantize model to lower data type. Currently supports FP8."""
        # pylint: disable=too-many-locals
        dataset = self.get_calib_dataset()
        model.eval()
        with self._try_offload_model(model):
            max_input_stats, _ = collect_stats(
                model,
                self.quant_config.device,
                dataset,
                cast(FP8QuantHook, self.hook).get_linear_layer_types(),
                percentile=self.quant_config.percentile,
                tqdm_desc="Collecting stats for Static Quantization.",
                batch_size=32,
            )
            for tf_quant_input in self.hook.iter_tf_quant_inputs(model):
                assert isinstance(tf_quant_input, HFTFQuantInputs)
                for quant_input in tf_quant_input.quant_inputs:
                    parent_module, local_names, names = (
                        quant_input.parent_module,
                        quant_input.local_names,
                        quant_input.target_names,
                    )
                    if isinstance(parent_module, torch.nn.ModuleList):
                        # for MoE model
                        # all module share same weight scale & act scale
                        parent_modules_w_local_name = []
                        for p_module in parent_module:
                            for local_name in local_names:
                                parent_modules_w_local_name.append(
                                    (p_module, local_name)
                                )

                        layers = [
                            p_module.get_submodule(local_name)
                            for p_module, local_name in parent_modules_w_local_name
                        ]

                        quant_results = self._get_weight_act_quantize_results(
                            model,
                            names,
                            max_input_stats,
                        )
                        q_layers = [
                            WeightActQuantizedLinearLayer.from_layer(
                                layer, quant_result
                            )
                            for layer, quant_result in zip(layers, quant_results)
                        ]
                        for (p_module, local_name), q_layer in zip(
                            parent_modules_w_local_name, q_layers
                        ):
                            setattr(p_module, local_name, q_layer)

                    else:
                        layers = [
                            parent_module.get_submodule(local_name)
                            for local_name in local_names
                        ]
                        quant_results = self._get_weight_act_quantize_results(
                            model,
                            names,
                            max_input_stats,
                        )
                        q_layers = [
                            WeightActQuantizedLinearLayer.from_layer(
                                layer, quant_result
                            )
                            for layer, quant_result in zip(layers, quant_results)
                        ]
                        for local_name, q_layer in zip(local_names, q_layers):
                            setattr(parent_module, local_name, q_layer)

        return model

    def convert(  # type: ignore[override]
        self,
        model: torch.nn.Module,
        convert_info_list: List[ConvertInfo],
    ) -> Generator[Tuple[str, Union[torch.Tensor, np.ndarray]], None, None]:
        """Convert Huggingface Model to Friendli format(.h5).

        Args:
            model (torch.nn.Module): Huggingface model.
            state_dict (Dict[str, torch.Tensor]):
                Dictionary of mapping of tensor name to tensor
            convert_info_list (List[ConvertInfo]):
                Dictionary of mapping converted params name to conversion functions.
                It will be depreciated.
        """
        quantized_param_names = []
        quantized_param_scale_names = []
        for tf_quant_input in self.hook.iter_tf_quant_inputs(model):
            assert isinstance(tf_quant_input, HFTFQuantInputs)
            for quant_input in tf_quant_input.quant_inputs:
                for target_name in quant_input.target_names:
                    quantized_param_names.append(f"{target_name}.weight")
                    quantized_param_scale_names.append(f"{target_name}.weight_scale")
                    quantized_param_scale_names.append(f"{target_name}.in_scale")

        self.pre_quantize(model)
        model = self.quantize(model)
        state_dict: Dict[str, torch.Tensor] = model.state_dict()

        with tqdm(total=len(state_dict), desc="Converting", unit="tensor") as pbar:
            for param_name, param in state_dict.items():
                param = param.to("cpu").detach()
                if param_name in quantized_param_names:
                    converted_param = param.to(torch.float8_e4m3fn).view(torch.int8)
                elif param_name in quantized_param_scale_names:
                    converted_param = param.to(torch.float32)
                else:
                    converted_param = param.to(
                        get_torch_data_type(self.converter.data_type)
                    )

                yield param_name, converted_param
                pbar.update()
