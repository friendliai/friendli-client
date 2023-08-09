# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Base."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Tuple, Type, cast

import datasets  # type: ignore[import]
import torch
from transformers import (  # type: ignore[import]
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

from periflow.enums import (
    CheckpointDataType,  # TODO: move this to periflow/modules/converter/enums.py
)
from periflow.errors import CheckpointQuantizationError
from periflow.modules.quantizer.formatter import (
    Int8QuantScale,
    Int8QuantScaleInput,
    Int8QuantScaleInputTuple,
    Int8QuantScaleResult,
    ModuleName,
)
from periflow.modules.quantizer.utils import (
    collect_max_stats,
    get_encoded_samples,
    get_int8_quant_scales,
    get_torch_data_type,
)


class AbstractQuantHook(ABC):
    """Quantization Hook for a specific model architecture."""

    def __init__(self, config: PretrainedConfig):
        """Initialize the Quantization Hook.

        Args:
            config (PreTrainedConfig): Hugging Face model configuration.
        """
        self.model_config = config

    @abstractmethod
    def get_quant_inputs(self, model: torch.nn.Module) -> Iterator[Int8QuantScaleInput]:
        """Returns the layers which should be quantized in transformer layers."""

    
    @property
    @abstractmethod
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer layer name."""

    def sort_qkv_output_stats(self, max_output_stat: torch.Tensor) -> torch.Tensor:
        """Sort max_output_stas for seperating qkv_layer's output_stats"""
        return max_output_stat

    def get_quant_result(
        self,
        quant_input: Int8QuantScaleInput,
        max_input_stats: Dict[ModuleName, torch.Tensor],
        max_output_stats: Dict[ModuleName, torch.Tensor],
    ) -> Int8QuantScaleResult:
        """Returns the quantization result of the quantized layer.

        If the model has another quantized layer, it should be implemented in the subclass. (
        """

        def get_scale(f: Int8QuantScaleInputTuple, sort_max_output_stats=False) -> Int8QuantScale:
            weight, name, start, end = f
            if sort_max_output_stats:
                return get_int8_quant_scales(
                    name,
                    max_input_stats[name],
                    weight[start:end],
                    self.sort_qkv_output_stats(max_output_stats[name])[start:end],
                )
            return get_int8_quant_scales(
                name,
                max_input_stats[name],
                weight[start:end],
                max_output_stats[name][start:end],
            )

        return Int8QuantScaleResult(
            layer_index=quant_input.layer_index,
            q=get_scale(quant_input.q, True),
            k=get_scale(quant_input.k, True),
            v=get_scale(quant_input.v, True),
            attn_fc=get_scale(quant_input.attn_fc),
            ff1=get_scale(quant_input.ff1),
            ff2=get_scale(quant_input.ff2),
        )


class SmoothQuantHook(AbstractQuantHook):
    """Quantization Hook for SmoothQuant."""

    @abstractmethod
    def get_smooth_norm_weights(
        self, model: torch.nn.Module
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm and linear layer's weight per transformer layer."""

    @abstractmethod
    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the type of linear layer (etc. qkv, linear layer) in transformer layer."""



class AbstractQuantizer(ABC):
    """Abstract Quantizer for a specific model architecture."""

    def __init__(self, hook: AbstractQuantHook, config: Dict[str, Any]):
        """Initialize the Quantizer.

        Args:
            hook (AbstractQuantHook): Quantization Hook for a specific model architecture
            config (Dict[str, Any]): Quantization configuration.
        """
        self.hook = hook
        self.config = config

    @abstractmethod
    def check_config(self) -> None:
        """Check if the quantization config is valid."""

    @abstractmethod
    def pre_quantize(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        data_type: CheckpointDataType,
    ) -> None:
        """Pre-procedure that should be called before quantize() is called."""

    @abstractmethod
    def quantize(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        data_type: CheckpointDataType,
    ) -> Iterator[Int8QuantScaleResult]:
        """Setting Quantizer from config and Quantize model."""


class SmoothQuantQuantizer(AbstractQuantizer):
    """Quantizer for SmoothQuant."""

    def check_config(self) -> None:
        """Check if the SmoothQuant quantization config is valid."""
        data_path_or_name = self.config["data_path_or_name"]
        if not os.path.exists(data_path_or_name):
            if data_path_or_name not in datasets.list_datasets():
                raise CheckpointQuantizationError(
                    f"Invalid dataset : {data_path_or_name}. "
                    "You can use one of the following datasets:"
                    "https://huggingface.co/datasets or a local dataset."
                )
        else:
            if self.config["data_format"] not in ["json", "csv", "parquet", "txt"]:
                raise CheckpointQuantizationError(
                    f"Invalid data_format : {self.config['data_format']}."
                    "You can use one of the following data formats: json, csv, parquet, or txt."
                )
            if self.config["data_split"] not in ["train", "validation", "test"]:
                raise CheckpointQuantizationError(
                    f"Invalid data_split : {self.config['data_split']}."
                    "You can use one of the following data splits: train, validation, or test."
                )
        try:
            torch.device(self.config["device"])
        except ValueError as err:
            raise CheckpointQuantizationError(
                f"Invalid device{self.config['device']}. {str(err)}"
            ) from err

        if (
            self.config["migration_strength"] < 0
            or self.config["migration_strength"] > 1
        ):
            raise CheckpointQuantizationError(
                f"Invalid migration_strength : {self.config['migration_strength']}."
                "You can use a value between 0 and 1."
            )

    # pylint: disable=too-many-locals
    @torch.no_grad()
    def _perform_smoothing(
        self,
        activation_norms: List[torch.Tensor],
        fc_weights: List[torch.Tensor],
        activation_max: torch.Tensor,
        *,
        migration_strength: float = 0.5,
        epsilon: float = 1e-5,
        inplace: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Perform activation-weight smoothing in SmoothQuant.

        Performs the activation-weight smoothing scheme described in SmoothQuant
        (Xiao et al., 2023), which migrates the amplitude of outliers from activations
        to weights of matmul layers. The function takes in the following parameters:

        Args:
            activation_norms: torch.Tensors representing affine parameters
                (i.e., beta and gamma) of a normalization layer before each matmul layer.
            fc_weights: torch.Tensors representing the weight matrices of the matmul layer.
            activation_max: The maximum activation value of inputs of the matmul layer.
            migration_strength: the strength of the activation migration. Default is 0.5.
            epsilon: The epsilon used for numerical stability when calculating the scales.
                Default is 1e-5.

        Returns:
            A tuple of three torch.Tensors: (smoothed_activation_norms, smoothed_fc_weights)

        The function calculates "scales" as `pow(|Activation|, migration_strength) /
        pow(|Weight|, 1-migration_strength)` and applies the smoothing effect into
        a normalization layer that exists before every matmul layer. This is done because
        it is more efficient than introducing a new smoothing layer before every matmul layer.
        Fusing the smoothing effect into the normalization layer results in a faster and
        more efficient implementation of the smoothing scheme.

        The function returns the smoothed normalization coefficients and the smoothed weight
        matrices after the smoothing process.
        """
        # shape of activation norms: [InChannels]
        # shape of fc weights: [OutChannels, InChannels]
        # shape of activation_max: [InChannels]

        assert activation_norms
        assert fc_weights

        assert activation_norms[0].ndim == 1
        in_channels = activation_norms[0].size(0)
        device = activation_norms[0].device
        dtype = activation_norms[0].dtype

        for norm in activation_norms:
            assert tuple(norm.size()) == (in_channels,)
            assert norm.device == device
            assert norm.dtype == dtype

        for weight in fc_weights:
            assert weight.ndim == 2
            assert weight.size(1) == in_channels
            assert weight.device == device
            assert weight.dtype == dtype

        activation_max = activation_max.to(device=device)
        weight_max = fc_weights[0].abs().max(dim=0).values
        for weight in fc_weights[1:]:
            weight_max = torch.maximum(weight_max, weight.abs().max(dim=0).values)

        assert tuple(activation_max.size()) == (in_channels,)
        assert tuple(weight_max.size()) == (in_channels,)
        alpha = migration_strength
        scales = (
            (
                activation_max.to(dtype=torch.float32).pow(alpha)
                / weight_max.to(dtype=torch.float32).pow(1 - alpha)
            )
            .clamp(min=epsilon)
            .to(dtype=dtype)
        )

        scaled_activation_norms = [act_norm / scales for act_norm in activation_norms]
        scaled_weights = [w * scales.view(1, -1) for w in fc_weights]

        if inplace:
            for dst, src in zip(activation_norms, scaled_activation_norms):
                dst.copy_(src)
            for dst, src in zip(fc_weights, scaled_weights):
                dst.copy_(src)

        return scaled_activation_norms, scaled_weights

    # pylint: enable=too-many-locals

    def smooth_inplace(
        self,
        model: torch.nn.Module,
        max_input_stats: Dict[ModuleName, torch.Tensor],
        migration_strength: float = 0.5,
    ):
        """Smooths the models in-place."""
        for norms, weights, name in cast(
            SmoothQuantHook, self.hook
        ).get_smooth_norm_weights(model):
            self._perform_smoothing(
                norms,
                weights,
                max_input_stats[name],
                migration_strength=migration_strength,
                inplace=True,
            )
    
    def smooth(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        data_type: CheckpointDataType,
    ) -> None:
        """Smooths the models before Quantization."""
        encoded_samples = get_encoded_samples(self.config, tokenizer)
        model.to(dtype=get_torch_data_type(data_type), device=self.config["device"])
        model.eval()
        max_input_stats, _ = collect_max_stats(
            model,
            encoded_samples,
            cast(SmoothQuantHook, self.hook).get_linear_layer_types(),
        )
        self.smooth_inplace(
            model,
            max_input_stats,
            migration_strength=self.config["migration_strength"],
        )

    def pre_quantize(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        data_type: CheckpointDataType,
    ) -> None:
        self.smooth(model, tokenizer, data_type)

    def quantize(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        data_type: CheckpointDataType,
    ) -> Iterator[Int8QuantScaleResult]:
        """Quantize model with SmoothQuant."""
        encoded_samples = get_encoded_samples(self.config, tokenizer)
        max_input_stats, max_output_stats = collect_max_stats(
            model,
            encoded_samples,
            cast(SmoothQuantHook, self.hook).get_linear_layer_types(),
            tqdm_desc="Collecting stats for Static Quantization.",
        )
        for quant_input in self.hook.get_quant_inputs(model):
            yield self.hook.get_quant_result(
                quant_input, max_input_stats, max_output_stats
            )
