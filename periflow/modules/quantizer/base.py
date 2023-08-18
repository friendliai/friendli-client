# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Base."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Tuple, Type, cast

import datasets  # type: ignore[import]
import torch
from transformers import PreTrainedTokenizerBase  # type: ignore[import]

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
    get_smoothquant_calibration_dataset,
    get_torch_data_type,
)


class AbstractQuantHook(ABC):
    """Quantization Hook for a specific model architecture."""

    @abstractmethod
    def get_quant_inputs(self, model: torch.nn.Module) -> Iterator[Int8QuantScaleInput]:
        """Returns the layers which should be quantized (etc. qkv, linear layer) in transformer layer."""

    @property
    @abstractmethod
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer layer name."""


class SmoothQuantHook(AbstractQuantHook):
    """Quantization Hook for SmoothQuant."""

    @abstractmethod
    def get_smooth_norm_weights(
        self, model: torch.nn.Module
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer layer."""

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

    @torch.no_grad()
    def _get_int8_quant_scales(
        self,
        layer_name: str,
        input_max: torch.Tensor,
        fc_weight: torch.Tensor,
        output_max: torch.Tensor,
    ) -> Int8QuantScale:
        """Get the quantization scales and int8 weight for a specific layer."""
        # shape of input_max: [InChannels]
        # shape of output_max: [OutChannels]
        # shape of fc_weight: [OutChannels, InChannels]
        assert input_max.ndim == 1
        assert output_max.ndim == 1

        in_channels = input_max.size(0)
        out_channels = output_max.size(0)
        assert tuple(fc_weight.size()) == (out_channels, in_channels)

        in_scale = 127.0 / float(input_max.detach().abs().max().item())
        weight_scale = 127.0 / float(fc_weight.detach().abs().max().item())
        out_scale = 127.0 / float(output_max.detach().abs().max().item())
        int8_weight = (
            (fc_weight.detach().float() * weight_scale)
            .round()
            .clip(-128, 127)
            .to(torch.int8)
            .cpu()
        )

        return Int8QuantScale(
            layer_name,
            in_scale=in_scale,
            weight_scale=weight_scale,
            out_scale=out_scale,
            int8_weight=int8_weight,
        )

    def _quantize(
        self,
        model: torch.nn.Module,
        max_input_stats: Dict[ModuleName, torch.Tensor],
        max_output_stats: Dict[ModuleName, torch.Tensor],
    ) -> Iterator[Int8QuantScaleResult]:
        """Get Quantization scales and int8 weight for all layers in the model."""

        def get_scale(f: Int8QuantScaleInputTuple) -> Int8QuantScale:
            weight, name, start, end = f
            return self._get_int8_quant_scales(
                name,
                max_input_stats[name],
                weight[start:end],
                max_output_stats[name][start:end],
            )

        for quant_input in self.hook.get_quant_inputs(model):
            yield Int8QuantScaleResult(
                layer_index=quant_input.layer_index,
                q=get_scale(quant_input.q),
                k=get_scale(quant_input.k),
                v=get_scale(quant_input.v),
                attn_fc=get_scale(quant_input.attn_fc),
                ff1=get_scale(quant_input.ff1),
                ff2=get_scale(quant_input.ff2),
            )

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
                    "You can use one of the following datasets: https://huggingface.co/datasets or a local dataset."
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
            )

        if (
            self.config["migration_strength"] < 0
            or self.config["migration_strength"] > 1
        ):
            raise CheckpointQuantizationError(
                f"Invalid migration_strength : {self.config['migration_strength']}."
                "You can use a value between 0 and 1."
            )

    @torch.no_grad()
    def _perform_smoothing(
        self,
        activation_norms: List[torch.Tensor],
        fc_weights: List[torch.Tensor],
        activation_max: torch.Tensor,
        migration_strength: float = 0.5,
        epsilon: float = 1e-5,
        inplace: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Perform activation-weight smoothing in SmoothQuant.

        Performs the activation-weight smoothing scheme described in SmoothQuant (Xiao et al., 2023), which migrates the
        amplitude of outliers from activations to weights of matmul layers. The function takes in the following parameters:

        Args:
            activation_norms: A list of torch.Tensors representing affine parameters (i.e., beta and gamma) of a normalization layer before each matmul layer.
            fc_weights: A list of torch.Tensors representing the weight matrices of the matmul layer.
            activation_max: A torch.Tensor representing the maximum activation value of inputs of the matmul layer.
            migration_strength: A float value representing the strength of the activation migration. Default is 0.5.
            epsilon: A float value representing the epsilon used for numerical stability when calculating the scales. Default is 1e-5.

        Returns:
            A tuple of three torch.Tensors: (smoothed_activation_norms, smoothed_fc_weights)

        The function calculates "scales" as `pow(|Activation|, migration_strength) / pow(|Weight|, 1-migration_strength)`
        and applies the smoothing effect into a normalization layer that exists before every matmul layer. This is done
        because it is more efficient than introducing a new smoothing layer before every matmul layer.
        Fusing the smoothing effect into the normalization layer results in a faster and
        more efficient implementation of the smoothing scheme.

        The function returns the smoothed normalization coefficients and the smoothed weight matrices after the smoothing process.
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
                norms, weights, max_input_stats[name], migration_strength, inplace=True
            )

    def quantize(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        data_type: CheckpointDataType,
    ) -> Iterator[Int8QuantScaleResult]:
        """Quantize model with SmoothQuant."""
        samples = get_smoothquant_calibration_dataset(
            data_path=self.config["data_path_or_name"],
            data_format=self.config["data_format"],
            data_split=self.config["data_split"],
            seed=self.config["seed"],
            num_samples=self.config["num_samples"],
        )
        encoded_samples = (
            tokenizer(
                x,
                return_tensors="pt",
                max_length=self.config["max_length"],
                truncation=True,
            ).input_ids
            for x in samples
        )
        model.to(dtype=get_torch_data_type(data_type), device=self.config["device"])
        model.eval()
        max_input_stats, max_output_stats = collect_max_stats(
            model,
            encoded_samples,
            cast(SmoothQuantHook, self.hook).get_linear_layer_types(),
        )
        self.smooth_inplace(
            model,
            max_input_stats,
            migration_strength=self.config["migration_strength"],
        )
        quant_results = self._quantize(model, max_input_stats, max_output_stats)

        return quant_results
