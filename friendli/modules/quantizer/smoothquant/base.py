# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli SmoothQuant Quantizer Base."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import fields
from typing import Any, Dict, Iterator, List, Tuple, cast

import datasets  # type: ignore[import]
import torch

from friendli.enums import CheckpointDataType
from friendli.errors import NotSupportedQuantConfigError
from friendli.modules.converter.base import DECODER_PREFIX
from friendli.modules.converter.interface import ModelConversionInterface
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.converter.utils import get_tokenizer
from friendli.modules.quantizer.base import AbstractQuantHook, CommonQuantizer
from friendli.modules.quantizer.layers import WeightActQuantizedLinearLayer
from friendli.modules.quantizer.schema.config import SmoothQuantConfig
from friendli.modules.quantizer.schema.data import (
    ModuleName,
    QuantInput,
    TFQuantInputs,
    TFQuantResults,
    WeightActQuantResult,
)
from friendli.modules.quantizer.utils import (
    collect_stats,
    get_weight_act_quant_scales,
    quantized_linear_weight_reshape,
    quantized_qkv_weight_reshape,
    safe_load_datasets,
    scale_reshape,
)


class PreSmoother(torch.nn.Module):
    """Module for containing smoothing scale.

    This module is used to contain the smoothing scale for the quantization.
    If the matmul layer have previous layer, the smoothing scale can be migrated
    to the previous layer. But, if the matmul layer is the first layer, the scale
    need to be stored in this module. Especially, When MLP ff2 layer with previous activation
    layer that prevent migrating the scale to the previous layer needs SmoothQuant, then,
    this module is used to store the smoothing scale. [SmoothQunat Issue #15]
    (https://github.com/mit-han-lab/smoothquant/issues/15#issuecomment-1353390283).

    Args:
        in_dim (float): input dimension of the matmul layer's weight dimension.
    """

    def __init__(self, in_dim: int):
        """Initialize PreSmoother."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(in_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of PreSmoother."""
        return (x * self.scale).to(x.dtype)


class SmoothQuantHook(AbstractQuantHook):
    """Quantization Hook for SmoothQuant."""

    @abstractmethod
    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the attention fc layer in the decoder block."""

    @abstractmethod
    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the second feed-forward layer in the decoder block."""

    @abstractmethod
    def iter_smooth_norm_weights(
        self, model: torch.nn.Module
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm and linear layer's weight per transformer block."""

    def _register_pre_smoother(self, linear: torch.nn.Linear) -> PreSmoother:
        """Register pre_smoother storing smoothing scale of linear layer."""
        pre_smoother = PreSmoother(linear.in_features).to(device=linear.weight.device)

        def pre_smoother_hook(_, x: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
            return (pre_smoother.forward(x[0]),)

        linear.register_forward_pre_hook(pre_smoother_hook)
        return pre_smoother

    def pre_smooth(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        """Pre-procedure for SmoothQuant before Smoothing."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args
        for decoder_layer in self.get_tf_blocks(model):
            if quant_args.attn_fc_smoothing:
                attn_fc_pre_smoother = self._register_pre_smoother(
                    self.get_attn_fc_layer(decoder_layer)
                )
                decoder_layer.add_module("attn_fc_pre_smoother", attn_fc_pre_smoother)
            if quant_args.ff2_smoothing:
                ff2_pre_smoother = self._register_pre_smoother(
                    self.get_ff2_layer(decoder_layer)
                )
                decoder_layer.add_module("ff2_pre_smoother", ff2_pre_smoother)
        return model

    def sort_qkv_output_stats(self, max_output_stat: torch.Tensor) -> torch.Tensor:
        """Sort max_output_stas for seperating qkv_layer's output_stats."""
        return max_output_stat

    def copy_norms(self, model: torch.nn.Module) -> torch.nn.Module:
        """Copy and Register norms in transformer block for seperated scaling.

        In some models(e.g. llama, gptj, codegen), matmul layers share activations
        from the same norms. Therefore, we need to copy and register the norms for
        seperated smoothing scale. For example, in llama, normalization layer is
        shared with gate linear layer and attention linear layer. Thus, we need to
        copy and register the norms for each linear layer and use them for smoothing.
        """
        return model

    def get_quant_result(
        self,
        quant_inputs: TFQuantInputs,
        **kwargs: Any,
    ) -> TFQuantResults:
        """Returns the quantization result of the quantized layer.

        If the model has another quantized layer, it should be implemented in the subclass.

        """
        max_input_stats: Dict[ModuleName, torch.Tensor] = kwargs["max_input_stats"]
        max_output_stats: Dict[ModuleName, torch.Tensor] = kwargs["max_output_stats"]

        def get_scale(
            quant_input: QuantInput,
        ) -> WeightActQuantResult:
            weight, name, start, end, sort_fn = (
                quant_input.weight,
                quant_input.name,
                quant_input.start_offset,
                quant_input.end_offset,
                quant_input.sort_fn,
            )

            return get_weight_act_quant_scales(
                name,
                max_input_stats[name],
                weight[start:end],
                sort_fn(max_output_stats[name])[start:end]
                if sort_fn
                else max_output_stats[name][start:end],
            )

        return TFQuantResults(
            layer_prefix_with_index=f"{self.quantized_layer_prefix}{quant_inputs.layer_index}.",
            block=quant_inputs.block,
            q=get_scale(quant_inputs.q),
            k=get_scale(quant_inputs.k),
            v=get_scale(quant_inputs.v),
            attn_fc=get_scale(quant_inputs.attn_fc),
            ff1=get_scale(quant_inputs.ff1),
            ff2=get_scale(quant_inputs.ff2),
        )

    @property
    def modified_layers_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for modified modules.

        This convert_info_list is used for modules that are modified for quantization.
        Especially, for attention fc layer and MLP ff2 layer, we need to migrate
        smooth scale to the previous layer. Thus, we add the smoothing scaler, and
        modify the convert_info_list for the modified modules.

        In some models, matmul layers share activations from the same norms. Therefore,
        we use `copy_norms()` to copy and register the norms for seperated smoothing scale.
        Thus, we modify the convert_info_list for the modified modules.
        """
        sq_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args
        new_layer_convert_info_list = []
        for i in range(self.converter.decoder_layer_num):
            layer_prefix = f"{self.quantized_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"

            if sq_args.attn_fc_smoothing:
                new_layer_convert_info_list.append(
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn_fc_pre_smoother.scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_proj/smoothquant/smoothing_vector:0",  # pylint: disable=line-too-long
                        reshape_fn=scale_reshape,
                    )
                )
            if sq_args.ff2_smoothing:
                new_layer_convert_info_list.append(
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff2_pre_smoother.scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}mlp/c_proj/smoothquant/smoothing_vector:0",  # pylint: disable=line-too-long
                        reshape_fn=scale_reshape,
                    )
                )

        return new_layer_convert_info_list

    @property
    def quantized_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for quantized layers."""
        convert_info_list = []
        for i in range(self.converter.decoder_layer_num):
            layer_prefix = f"{self.quantized_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}q.weight_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_attn/smoothquant/q_weight_scale:0",  # pylint: disable=line-too-long
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}k.weight_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_attn/smoothquant/k_weight_scale:0",  # pylint: disable=line-too-long
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}v.weight_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_attn/smoothquant/v_weight_scale:0",  # pylint: disable=line-too-long
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}q.out_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_attn/smoothquant/q_out_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}k.out_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_attn/smoothquant/k_out_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}v.out_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_attn/smoothquant/v_out_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}q.in_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_attn/smoothquant/in_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn_fc.weight_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_proj/smoothquant/weight_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn_fc.out_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_proj/smoothquant/out_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn_fc.in_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_proj/smoothquant/in_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff1.weight_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}mlp/c_fc/smoothquant/weight_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff1.out_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}mlp/c_fc/smoothquant/out_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff1.in_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}mlp/c_fc/smoothquant/in_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff2.weight_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}mlp/c_proj/smoothquant/weight_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff2.out_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}mlp/c_proj/smoothquant/out_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff2.in_scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}mlp/c_proj/smoothquant/in_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}q.weight",
                            f"{layer_prefix}k.weight",
                            f"{layer_prefix}v.weight",
                        ],
                        data_type=CheckpointDataType.INT8,
                        converted_name=f"{converted_prefix}attn/c_attn/smoothquant/weight:0",
                        reshape_fn=quantized_qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn_fc.weight"],
                        data_type=CheckpointDataType.INT8,
                        converted_name=f"{converted_prefix}attn/c_proj/smoothquant/weight:0",
                        reshape_fn=quantized_linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff1.weight"],
                        data_type=CheckpointDataType.INT8,
                        converted_name=f"{converted_prefix}mlp/c_fc/smoothquant/weight:0",
                        reshape_fn=quantized_linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff2.weight"],
                        data_type=CheckpointDataType.INT8,
                        converted_name=f"{converted_prefix}mlp/c_proj/smoothquant/weight:0",
                        reshape_fn=quantized_linear_weight_reshape,
                    ),
                ]
            )
        return convert_info_list


class SmoothQuantQuantizer(CommonQuantizer, ModelConversionInterface):
    """Quantizer for SmoothQuant."""

    def check_config(self) -> None:
        """Check if the SmoothQuant quantization config is valid."""
        quant_config = cast(SmoothQuantConfig, self.quant_config)
        smoothquant_args = quant_config.smoothquant_args
        super().check_config()
        if 0 > smoothquant_args.migration_strength > 1:
            raise NotSupportedQuantConfigError(
                invalid_option=str(smoothquant_args.migration_strength),
                valid_options=["between 0 and 1."],
            )

    def get_calib_dataset(self) -> datasets.Dataset:
        """Get calibration dataset for SmoothQuant."""
        data_cfg = self.quant_config.calibration_dataset
        tokenizer = get_tokenizer(self.converter.config.name_or_path)
        dataset = safe_load_datasets(data_cfg)

        def preprocess(example) -> Dict[str, torch.Tensor]:
            truncate_length = data_cfg.max_length * 4
            while True:
                input_ids = tokenizer(
                    example[data_cfg.lookup_column_name][:truncate_length],
                    return_tensors="pt",
                    max_length=data_cfg.max_length * 2,
                    truncation=True,
                    padding=False,
                ).input_ids

                if input_ids.size(
                    1
                ) >= data_cfg.max_length * 2 or truncate_length >= len(
                    example[data_cfg.lookup_column_name]
                ):
                    input_ids = input_ids[:, : data_cfg.max_length]
                    break

                truncate_length *= 2
            return {"input_ids": input_ids}

        dataset = (
            dataset.shuffle(self.quant_config.seed)
            .select(range(data_cfg.num_samples))
            .select_columns([data_cfg.lookup_column_name])
            .map(function=preprocess)
        )

        return dataset

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

        # pylint: disable=too-many-locals
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

    def _smooth(
        self,
        model: torch.nn.Module,
    ) -> None:
        """Smooths the models before Quantization."""
        model.to(device=torch.device(self.quant_config.device))
        model.eval()
        model = cast(SmoothQuantHook, self.hook).pre_smooth(model)

        # collect stats for SmoothQuant scale.
        dataset = self.get_calib_dataset()
        quant_config = cast(SmoothQuantConfig, self.quant_config)
        max_input_stats, _ = collect_stats(
            model,
            quant_config.device,
            dataset,
            cast(SmoothQuantHook, self.hook).get_linear_layer_types(),
            tqdm_desc="Collecting stats for Smoothing.",
            percentile=100.0,
        )

        # TODO change name to pre_act_params, post_act_params
        # (attn_fc, ff2 are not scaled with norms)
        for norms, weights, name in cast(
            SmoothQuantHook, self.hook
        ).iter_smooth_norm_weights(model):
            self._perform_smoothing(
                norms,
                weights,
                max_input_stats[name],
                migration_strength=quant_config.smoothquant_args.migration_strength,
                inplace=True,
            )

    def pre_quantize(
        self,
        model: torch.nn.Module,
    ) -> None:
        """Pre-procedure that should be called before quantize() is called."""
        self._smooth(model)

    def quantize(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        """Quantize model with SmoothQuant."""
        dataset = self.get_calib_dataset()
        max_input_stats, max_output_stats = collect_stats(
            model,
            self.quant_config.device,
            dataset,
            cast(SmoothQuantHook, self.hook).get_linear_layer_types(),
            percentile=self.quant_config.percentile,
            tqdm_desc="Collecting stats for Static Quantization.",
        )
        for quant_input in self.hook.iter_tf_quant_inputs(model):
            quant_result = cast(SmoothQuantHook, self.hook).get_quant_result(
                quant_input,
                max_input_stats=max_input_stats,
                max_output_stats=max_output_stats,
            )

            for field in fields(quant_result):
                layer_quant_result = getattr(quant_result, field.name)
                if isinstance(layer_quant_result, WeightActQuantResult):
                    layer = model.get_submodule(layer_quant_result.module_name)
                    q_layer = WeightActQuantizedLinearLayer.from_layer(
                        layer, layer_quant_result
                    )
                    quant_result.block.add_module(field.name, q_layer)

        return model
