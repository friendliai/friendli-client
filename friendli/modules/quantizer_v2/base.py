# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Model Quantization Interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple, Type

import huggingface_hub  # type: ignore
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel  # type: ignore

from friendli.errors import NotSupportedQuantConfigError
from friendli.logging import logger
from friendli.modules.quantizer_v2.enums import QuantDatasetFormat
from friendli.modules.quantizer_v2.layers import (
    WeightActQuantizedLinearLayer,
    WeightOnlyQuantizedLinearLayer,
)
from friendli.modules.quantizer_v2.schema.config import OneOfQuantConfig
from friendli.modules.quantizer_v2.schema.data import TFQuantInputs
from friendli.modules.quantizer_v2.utils import (
    collect_stats,
    get_weight_act_quant_scales,
    get_weight_only_quant_scales,
    offload_module_sequence,
    send_model_to_device,
)


class AbstractQuantHookV2(ABC):
    """Abstract Quantization Hook for a specific model."""

    def __init__(self, quant_config: OneOfQuantConfig, model_config: PretrainedConfig):
        """Initialize the Quantization Hook.

        Args:
            quant_config (OneOfQuantConfig): Quantization configuration.
            model_config (PretrainedConfig): Model configuration.
        """
        self.quant_config = quant_config
        self.model_config = model_config

    @abstractmethod
    def check_model_config(self) -> None:
        """Check if the model is quantizable."""

    @abstractmethod
    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module], ...]:
        """Get linear layer types in the model."""

    @abstractmethod
    def get_tf_blocks(self, model: PreTrainedModel) -> List[torch.nn.Module]:
        """Get tensor fusion blocks in the model."""

    @abstractmethod
    def iter_tf_quant_inputs(self, model: PreTrainedModel) -> Iterator[TFQuantInputs]:
        """Iterate over TFQuantInputs."""

    @property
    @abstractmethod
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block name."""


class AbstractQuantizerV2(ABC):
    """Abstract class for quantizer."""

    def __init__(self, hook: AbstractQuantHookV2, config: OneOfQuantConfig):
        """Initialize AbstractQuantizer."""
        self.config = config
        self.hook = hook

    def check_config(self) -> None:
        """Check if the model is quantizable."""
        self.hook.check_model_config()
        calibration_dataset_config = self.config.calibration_dataset
        data_path_or_name = calibration_dataset_config.path_or_name
        percentile = self.config.percentile
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
            torch.device(self.config.device)
        except ValueError as err:
            raise NotSupportedQuantConfigError(
                invalid_option=self.config.device,
                valid_options=["cpu", "cuda"],
            ) from err

    @contextmanager
    def _try_offload_model(self, model: PreTrainedModel):
        if not self.config.offload:
            logger.info("Offloading not enabled. Skipping.")
            model.to(self.config.device)
            yield
        else:
            logger.info("Offloading enabled.")
            tf_blocks = self.hook.get_tf_blocks(model)
            send_model_to_device(model, self.config.device, exclude=tf_blocks)
            with offload_module_sequence(tf_blocks, self.config.device):
                yield

    @abstractmethod
    def quantize(self, model: PreTrainedModel) -> PreTrainedModel:
        """Quantize model."""

    def pre_quantize(self, model: PreTrainedModel) -> PreTrainedModel:
        """Preprocess model before quantization."""

    def post_quantize(self, model: PreTrainedModel) -> PreTrainedModel:
        """Postprocess model after quantization."""

    @abstractmethod
    def get_quant_config(self) -> Dict[str, Any]:
        """Get quantizer config."""


class AbstractWeightOnlyQuantizer(AbstractQuantizerV2):
    """Abstract class for weight only quantizer."""

    def quantize(self, model: PreTrainedModel) -> PreTrainedModel:
        """Return quantized model."""
        with self._try_offload_model(model):
            for tf_quant_inputs in tqdm(
                self.hook.iter_tf_quant_inputs(model),
                total=len(self.hook.get_tf_blocks(model)),
                desc="Quantize model..",
            ):
                for quant_input in tf_quant_inputs.quant_inputs:
                    parent_module, local_names, names = (
                        quant_input.parent_module,
                        quant_input.local_names,
                        quant_input.target_names,
                    )
                    parent_modules_w_local_name = []
                    if isinstance(parent_module, torch.nn.ModuleList):
                        # For MoE models with seperate expert layers
                        for p_module in parent_module:
                            for local_name in local_names:
                                parent_modules_w_local_name.append(
                                    (p_module, local_name)
                                )
                    else:
                        assert isinstance(parent_module, torch.nn.Module)
                        for local_name in local_names:
                            parent_modules_w_local_name.append(
                                (parent_module, local_name)
                            )
                    layers = [
                        p_module.get_submodule(local_name)
                        for p_module, local_name in parent_modules_w_local_name
                    ]
                    assert self.config.quant_scale_dtype
                    quant_results = get_weight_only_quant_scales(
                        model,
                        names,
                        quant_dtype=self.config.quant_dtype,
                        quant_scale_dtype=self.config.quant_scale_dtype,
                        q_group_size=self.config.quant_group_size,
                        use_symmetric=self.config.use_symmetric,
                    )
                    q_layers = [
                        WeightOnlyQuantizedLinearLayer.from_layer(layer, quant_result)
                        for layer, quant_result in zip(layers, quant_results)
                    ]
                    for (p_module, local_name), q_layer in zip(
                        parent_modules_w_local_name, q_layers
                    ):
                        setattr(p_module, local_name, q_layer)
        return model


class AbstractWeightActQuantizer(AbstractQuantizerV2):
    """Abstract class for weight and activation quantizer."""

    @abstractmethod
    def get_calib_dataloader(self) -> DataLoader:
        """Get encoded calibration dataset."""

    def quantize(self, model: PreTrainedModel) -> PreTrainedModel:
        """Return quantized model."""
        with self._try_offload_model(model):
            max_input_stats, _ = collect_stats(
                model,
                self.config.device,
                self.get_calib_dataloader(),
                self.hook.get_linear_layer_types(),
                percentile=self.config.percentile,
                tqdm_desc="Collecting stats for Static Quantization.",
            )
            for tf_quant_inputs in tqdm(
                self.hook.iter_tf_quant_inputs(model),
                total=len(self.hook.get_tf_blocks(model)),
                desc="Quantize model..",
            ):
                for quant_input in tf_quant_inputs.quant_inputs:
                    parent_module, local_names, names = (
                        quant_input.parent_module,
                        quant_input.local_names,
                        quant_input.target_names,
                    )
                    parent_modules_w_local_name = []
                    if isinstance(parent_module, torch.nn.ModuleList):
                        # For MoE models with seperate expert layers
                        for p_module in parent_module:
                            for local_name in local_names:
                                parent_modules_w_local_name.append(
                                    (p_module, local_name)
                                )
                    else:
                        assert isinstance(parent_module, torch.nn.Module)
                        for local_name in local_names:
                            parent_modules_w_local_name.append((p_module, local_name))
                    layers = [
                        p_module.get_submodule(local_name)
                        for p_module, local_name in parent_modules_w_local_name
                    ]
                    assert self.config.quant_scale_dtype
                    quant_results = get_weight_act_quant_scales(
                        model,
                        names,
                        max_input_stats,
                        quant_scale_dtype=self.config.quant_scale_dtype,
                        quant_dtype=self.config.quant_dtype,
                    )
                    q_layers = [
                        WeightActQuantizedLinearLayer.from_layer(layer, quant_result)
                        for layer, quant_result in zip(layers, quant_results)
                    ]
                    for (p_module, local_name), q_layer in zip(
                        parent_modules_w_local_name, q_layers
                    ):
                        setattr(p_module, local_name, q_layer)
        return model
