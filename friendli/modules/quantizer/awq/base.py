# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli AWQ Quantizer Base."""

from __future__ import annotations

import gc
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict, Iterator, List, Tuple, Type, cast

import datasets  # type: ignore[import]
import torch
from datasets.utils.logging import disable_progress_bar  # type: ignore[import]
from tqdm import tqdm

from friendli.enums import CheckpointDataType
from friendli.errors import QuantizationError
from friendli.logging import logger
from friendli.modules.converter.base import DECODER_PREFIX
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.converter.utils import get_tokenizer
from friendli.modules.quantizer.awq.utils import (
    apply_module_clip,
    apply_module_scale,
    search_module_clip,
    search_module_scale,
)
from friendli.modules.quantizer.base import AbstractQuantHook, CommonQuantizer
from friendli.modules.quantizer.layers import WeightOnlyQuantizedLinearLayer
from friendli.modules.quantizer.schema.config import AWQConfig
from friendli.modules.quantizer.schema.data import (
    ModuleName,
    QuantInput,
    TFQuantInputs,
    TFQuantResults,
    WeightOnlyQuantResult,
)
from friendli.modules.quantizer.utils import (
    collect_inps,
    get_weight_only_quant_scales,
    offload_module_sequence,
    quantized_linear_weight_reshape,
    quantized_qkv_weight_reshape,
    safe_load_datasets,
    scale_reshape,
    send_model_to_device,
)


class AWQScaler(torch.nn.Module):
    """Store AWQ scale before linear layers.

    If the linear layer is quantized, but the previous layer can't be scaled,
    then we need to store the AWQ scale in a separate module. This module
    is used to store the AWQ scale.
    """

    def __init__(self, in_dim: int):
        """Initialize AWQScaler."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        """Scale input by AWQ scale."""
        return (x / self.scale.view(1, 1, -1)).to(x.dtype)


class AWQHook(AbstractQuantHook):
    """Quantization Hook for AWQ."""

    @abstractmethod
    def iter_inspect_modules(
        self,
        block: torch.nn.Module,
    ) -> Iterator[
        Tuple[
            List[torch.nn.Module],
            List[Tuple[ModuleName, torch.nn.Linear]],
            torch.nn.Module,
            ModuleName,
        ]
    ]:
        """Returns iterator of modules to inspect for AWQ scale."""

    @abstractmethod
    def add_pre_scaler(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        """Add scaler for storing AWQ scale in modules."""

    @abstractmethod
    def get_inspect_module_types(
        self, block: torch.nn.Module
    ) -> Tuple[Type[torch.nn.Module], ...]:
        """Returns the type of inspect modules in transformer block."""

    def _register_pre_scaler(
        self,
        linear: torch.nn.Module,
    ) -> AWQScaler:
        """Register pre-scaler for storing AWQ scale in modules."""
        scaler = AWQScaler(linear.in_features)  # type: ignore

        def pre_scaler_hook(_, x: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
            return (scaler(x[0]),)

        linear.register_forward_pre_hook(pre_scaler_hook)
        return scaler

    def get_quant_result(
        self,
        quant_inputs: TFQuantInputs,
        **kwargs: Any,
    ) -> TFQuantResults:
        """Get quantization result for AWQ."""
        awq_config = cast(AWQConfig, self.quant_config)

        def get_scale(
            quant_input: QuantInput,
        ) -> WeightOnlyQuantResult:
            weight, name, start, end = (
                quant_input.weight,
                quant_input.name,
                quant_input.start_offset,
                quant_input.end_offset,
            )
            weight = weight.to(awq_config.device)

            return get_weight_only_quant_scales(
                layer_name=name,
                w=weight[start:end],
                q_bit=awq_config.awq_args.quant_bit,
                q_group_size=awq_config.awq_args.quant_group_size,
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
    def quant_dtype(self) -> CheckpointDataType:
        """Return the quantization dtype."""
        quant_config = cast(AWQConfig, self.quant_config)
        awq_args = quant_config.awq_args
        if awq_args.quant_bit == 4:
            return CheckpointDataType.INT4
        return CheckpointDataType.INT8

    @property
    @abstractmethod
    def avoid_clipping_layer_names(self) -> List[str]:
        """Return the layer names to avoid clipping."""

    @property
    @abstractmethod
    def modified_layers_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for modified layers."""

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
                        param_names=[
                            f"{layer_prefix}q.weight_scale",
                            f"{layer_prefix}k.weight_scale",
                            f"{layer_prefix}v.weight_scale",
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/awq/scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}q.zeros",
                            f"{layer_prefix}k.zeros",
                            f"{layer_prefix}v.zeros",
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/awq/zero:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}q.weight",
                            f"{layer_prefix}k.weight",
                            f"{layer_prefix}v.weight",
                        ],
                        data_type=self.quant_dtype,
                        converted_name=f"{converted_prefix}attn/c_attn/awq/weight:0",
                        reshape_fn=quantized_qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn_fc.weight_scale"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/awq/scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn_fc.zeros"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/awq/zero:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn_fc.weight"],
                        data_type=self.quant_dtype,
                        converted_name=f"{converted_prefix}attn/c_proj/awq/weight:0",
                        reshape_fn=quantized_linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff1.weight_scale"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/awq/scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff1.zeros"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/awq/zero:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff1.weight"],
                        data_type=self.quant_dtype,
                        converted_name=f"{converted_prefix}mlp/c_fc/awq/weight:0",
                        reshape_fn=quantized_linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff2.weight_scale"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/awq/scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff2.zeros"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/awq/zero:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff2.weight"],
                        data_type=self.quant_dtype,
                        converted_name=f"{converted_prefix}mlp/c_proj/awq/weight:0",
                        reshape_fn=quantized_linear_weight_reshape,
                    ),
                ]
            )
        return convert_info_list


class AWQQuantizer(CommonQuantizer):
    """Quantizer for AWQ."""

    def check_config(self) -> None:
        """Check if the AWQ quantization config is valid."""
        super().check_config()
        quant_config = cast(AWQConfig, self.quant_config)
        awq_args = quant_config.awq_args
        if awq_args.quant_bit not in [4, 8]:
            raise QuantizationError(
                f"Invalid quant_bit {awq_args.quant_bit} for AWQ."
                "You can only use 4 or 8 bit for AWQ."
            )
        if awq_args.quant_group_size not in [64]:
            raise QuantizationError(
                f"Invalid quant_group_size {awq_args.quant_group_size} for AWQ."
                "You can only use 64 for AWQ."
            )

    def get_calib_dataset(self) -> datasets.Dataset:
        """Get calibration dataset for AWQ."""
        data_cfg = self.quant_config.calibration_dataset
        tokenizer = get_tokenizer(self.converter.config.name_or_path)
        dataset = safe_load_datasets(data_cfg)

        def preprocess(sample) -> Dict[str, Any]:
            """Preprocess dataset for AWQ."""
            return {"input_ids": tokenizer(sample).input_ids}

        disable_progress_bar()
        dataset = (
            dataset.shuffle(self.quant_config.seed)
            .filter(
                lambda sample: len(tokenizer(sample)) <= data_cfg.max_length,
                input_columns=data_cfg.lookup_column_name,
            )
            .map(function=preprocess, input_columns=data_cfg.lookup_column_name)
            .filter(
                lambda sample: torch.tensor(sample).numel() != 0,
                input_columns="input_ids",
            )
            .select(range(data_cfg.num_samples))
        )

        return dataset

    def get_batched_samples(self):
        """Get batched samples from dataset."""
        dataset = self.get_calib_dataset()

        samples = []
        for sample in dataset["input_ids"]:
            samples.append(torch.tensor(sample))

        seqlen = self.quant_config.calibration_dataset.max_length
        batched_samples = torch.cat(samples)
        if len(batched_samples) // seqlen == 0:
            return batched_samples.unsqueeze(0)

        batched_samples = [
            batched_samples[i * seqlen : (i + 1) * seqlen].unsqueeze(0)
            for i in range(len(batched_samples) // seqlen)
        ]
        batched_samples = torch.cat(batched_samples, dim=0)

        return batched_samples

    def _apply_awq_scale_clip_block(
        self,
        block: torch.nn.Module,
        block_args: Tuple[Any, ...],
        block_kwargs: Dict[str, Any],
    ) -> None:
        """Search AWQ scale, clipping range and Apply them into a transformer block."""
        # pylint: disable=too-many-locals

        inpsected_mod_types = cast(AWQHook, self.hook).get_inspect_module_types(block)
        args_dict, kwargs_dict = collect_inps(
            block,
            block_args,
            block_kwargs,
            self.quant_config.device,
            tuple([*self.hook.get_linear_layer_types(), *inpsected_mod_types]),
        )
        awq_args = cast(AWQConfig, self.quant_config).awq_args
        for prev_ops, linear_tuples, module2inspect, module2inspect_name in cast(
            AWQHook, self.hook
        ).iter_inspect_modules(block):
            linear_inp = args_dict[linear_tuples[0][0]][0]
            linear_layers = [linear for _, linear in linear_tuples]

            scales = search_module_scale(
                module2inspect,
                args_dict[module2inspect_name],
                kwargs_dict[module2inspect_name],
                linear_layers,
                linear_inp,
                awq_args.quant_group_size,
                awq_args.quant_bit,
            )

            apply_module_scale(
                prev_ops,
                linear_layers,
                scales.to(self.quant_config.device),
            )

            for name, _ in linear_tuples:
                assert len(args_dict[name]) == 1
                assert torch.equal(args_dict[name][0], linear_inp)
                args_dict[name] = (args_dict[name][0].div(scales.view(1, -1)),)

        named_linears = {
            name: m
            for name, m in block.named_modules()
            if isinstance(m, torch.nn.Linear)
        }
        for name, linear in named_linears.items():
            if any(
                (
                    avoid in name
                    for avoid in cast(AWQHook, self.hook).avoid_clipping_layer_names
                )
            ):
                continue
            max_val = search_module_clip(
                linear.weight,
                args_dict[name][0],
                awq_args.quant_group_size,
                awq_args.quant_bit,
                n_sample_token=self.quant_config.calibration_dataset.num_samples,
            )
            apply_module_clip(
                max_val.to(self.quant_config.device),
                linear,
            )

    def get_input_kwargs_tf_blocks(
        self,
        model: torch.nn.Module,
    ) -> Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]:
        """Gather input tensor and kwargs from the designated pytorch module."""
        block_args = []
        block_kwargs = []

        num_tf_blocks = len(self.hook.get_tf_blocks(model))
        progress_bar = tqdm(
            range(num_tf_blocks),
            total=num_tf_blocks,
            desc="Collect args for transformer blocks..",
        )

        def hook(m, args, kwargs):  # pylint: disable=unused-argument
            block_args.append(
                tuple(
                    (t.detach().cpu() if isinstance(t, torch.Tensor) else t)
                    for t in args
                )
            )
            block_kwargs.append(
                {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in kwargs.items()
                }
            )
            progress_bar.update()

        removables = []
        for tf_block in self.hook.get_tf_blocks(model):
            removables.append(
                tf_block.register_forward_pre_hook(hook, with_kwargs=True)
            )

        batched_samples = self.get_batched_samples()
        model(batched_samples.to(self.quant_config.device), use_cache=False)

        for removable in removables:
            removable.remove()

        return block_args, block_kwargs

    def get_attributes(self) -> Dict[str, Any]:
        """Return the attributes of the converted model."""
        attributes = self.converter.get_attributes()
        awq_args = cast(AWQConfig, self.quant_config).awq_args
        attributes["quant_scheme"] = self.quant_config.mode.value  # awq
        attributes["quant_group_size"] = awq_args.quant_group_size
        attributes["quant_bit"] = awq_args.quant_bit
        return attributes

    @contextmanager
    def _try_offload_model(self, model: torch.nn.Module):
        if not self.quant_config.offload:
            logger.info("AWQ offloading not enabled. Skipping.")
            model.to(self.quant_config.device)
            yield
        else:
            logger.info("AWQ offloading enabled.")
            tf_blocks = self.hook.get_tf_blocks(model)
            send_model_to_device(model, self.quant_config.device, exclude=tf_blocks)
            with offload_module_sequence(tf_blocks, self.quant_config.device):
                yield

    @torch.no_grad()
    def _apply_awq_scale_clip(
        self,
        model: torch.nn.Module,
    ) -> None:
        """Search AWQ scale, clipping range and Apply them into model."""
        # pylint: disable=too-many-locals
        model.eval()
        with self._try_offload_model(model):
            tf_blocks = self.hook.get_tf_blocks(model)
            block_args, block_kwargs = self.get_input_kwargs_tf_blocks(model)

            gc.collect()
            torch.cuda.empty_cache()

            for block, args, kwargs in tqdm(
                zip(
                    tf_blocks,
                    block_args,
                    block_kwargs,
                ),
                total=len(tf_blocks),
                desc="Search and Apply AWQ Scale, Clip range..",
            ):
                self._apply_awq_scale_clip_block(block, args, kwargs)
                gc.collect()
                torch.cuda.empty_cache()

    @torch.no_grad()
    def pre_quantize(
        self,
        model: torch.nn.Module,
    ) -> None:
        """Pre-procedure that should be called before quantize() is called."""
        model = cast(AWQHook, self.hook).add_pre_scaler(model)
        self._apply_awq_scale_clip(model)

    @torch.no_grad()
    def quantize(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        """Quantize model with AWQ."""
        model.eval()
        for quant_input in tqdm(
            self.hook.iter_tf_quant_inputs(model),
            total=len(self.hook.get_tf_blocks(model)),
            desc="Quantize model..",
        ):
            quant_result = cast(AWQHook, self.hook).get_quant_result(
                quant_input, quant_config=cast(AWQConfig, self.quant_config)
            )
            for field in fields(quant_result):
                layer_quant_result = getattr(quant_result, field.name)
                if isinstance(layer_quant_result, WeightOnlyQuantResult):
                    layer = model.get_submodule(layer_quant_result.module_name)
                    q_layer = WeightOnlyQuantizedLinearLayer.from_layer(
                        layer, layer_quant_result
                    )
                    quant_result.block.add_module(field.name, q_layer)

        return model
