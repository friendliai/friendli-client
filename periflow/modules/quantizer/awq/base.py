# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow AWQ Quantizer Base."""

from __future__ import annotations

import gc
from abc import abstractmethod
from dataclasses import fields
from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, cast

import datasets  # type: ignore[import]
import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar  # type: ignore[import]
from tqdm import tqdm

from periflow.errors import QuantizationError
from periflow.modules.converter.utils import get_tokenizer, nontype_partial
from periflow.modules.quantizer.awq.utils import (
    apply_module_clip,
    apply_module_scale,
    search_module_clip,
    search_module_scale,
)
from periflow.modules.quantizer.base import AbstractQuantHook, CommonQuantizer
from periflow.modules.quantizer.schema.config import AWQConfig
from periflow.modules.quantizer.schema.data import (
    ModuleName,
    QuantInput,
    TFQuantInputs,
    TFQuantResults,
    WeightOnlyQuantResult,
)
from periflow.modules.quantizer.utils import (
    collect_inps,
    get_weight_only_quant_scales,
    quantized_linear_weight_convert,
    quantized_qkv_weight_convert,
    safe_load_datasets,
    scale_convert,
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
    """Quantization Hook for SmoothQuant."""

    @abstractmethod
    def iter_inspect_modules(
        self, model: torch.nn.Module, model_kwargs: Dict[str, Any]
    ) -> Iterator[
        Tuple[
            List[torch.nn.Module],
            List[torch.nn.Module],
            ModuleName,
            torch.nn.Module,
            Dict[str, Any],
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
        quant_input: TFQuantInputs,
        **kwargs: Any,
    ) -> TFQuantResults:
        """Get quantization result for AWQ."""
        awq_args = cast(AWQConfig, self.quant_config).awq_args

        def get_scale(
            quant_input: QuantInput,
        ) -> WeightOnlyQuantResult:
            weight, name, start, end = (
                quant_input.weight,
                quant_input.name,
                quant_input.start_offset,
                quant_input.end_offset,
            )

            return get_weight_only_quant_scales(
                layer_name=name,
                w=weight[start:end],
                q_bit=awq_args.quant_bit,
                q_group_size=awq_args.quant_group_size,
            )

        return TFQuantResults(
            layer_prefix_with_index=f"{self.quantized_layer_prefix}{quant_input.layer_index}.",
            q=get_scale(quant_input.q),
            k=get_scale(quant_input.k),
            v=get_scale(quant_input.v),
            attn_fc=get_scale(quant_input.attn_fc),
            ff1=get_scale(quant_input.ff1),
            ff2=get_scale(quant_input.ff2),
        )

    @property
    @abstractmethod
    def avoid_clipping_layer_names(self) -> List[str]:
        """Return the layer names to avoid clipping."""

    @property
    def modified_layers_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for modified layers."""
        return {
            "attn/c_proj/awq/pre_scale:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".attn.scaler.scale"],
                data_type="fp32",
            ),
            "mlp/c_proj/awq/pre_scale:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".ffn.scaler.scale"],
                data_type="fp32",
            ),
        }

    @property
    def quantized_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for quantized layers."""
        quant_config = cast(AWQConfig, self.quant_config)
        n_bit = quant_config.awq_args.quant_bit
        return {
            "attn/c_attn/awq/scale:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[
                    ".q.woq_weight_scale",
                    ".k.woq_weight_scale",
                    ".v.woq_weight_scale",
                ],
                data_type=self.converter.data_type,
            ),
            "attn/c_attn/awq/zero:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[
                    ".q.woq_weight_zp",
                    ".k.woq_weight_zp",
                    ".v.woq_weight_zp",
                ],
                data_type=self.converter.data_type,
            ),
            "attn/c_attn/awq/weight:0": nontype_partial(
                quantized_qkv_weight_convert,
                per_layer_postfixes=[
                    ".q.woq_weight",
                    ".k.woq_weight",
                    ".v.woq_weight",
                ],
                n_bit=n_bit,
            ),
            "attn/c_proj/awq/scale:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".attn_fc.woq_weight_scale"],
                data_type=self.converter.data_type,
            ),
            "attn/c_proj/awq/zero:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".attn_fc.woq_weight_zp"],
                data_type=self.converter.data_type,
            ),
            "attn/c_proj/awq/weight:0": nontype_partial(
                quantized_linear_weight_convert,
                per_layer_postfixes=[".attn_fc.woq_weight"],
                n_bit=n_bit,
            ),
            "mlp/c_fc/awq/scale:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".ff1.woq_weight_scale"],
                data_type=self.converter.data_type,
            ),
            "mlp/c_fc/awq/zero:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".ff1.woq_weight_zp"],
                data_type=self.converter.data_type,
            ),
            "mlp/c_fc/awq/weight:0": nontype_partial(
                quantized_linear_weight_convert,
                per_layer_postfixes=[".ff1.woq_weight"],
                n_bit=n_bit,
            ),
            "mlp/c_proj/awq/scale:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".ff2.woq_weight_scale"],
                data_type=self.converter.data_type,
            ),
            "mlp/c_proj/awq/zero:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".ff2.woq_weight_zp"],
                data_type=self.converter.data_type,
            ),
            "mlp/c_proj/awq/weight:0": nontype_partial(
                quantized_linear_weight_convert,
                per_layer_postfixes=[".ff2.woq_weight"],
                n_bit=n_bit,
            ),
        }


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
        block_inps: torch.Tensor,
        block_kwargs: Dict[str, Any],
    ) -> None:
        """Search AWQ scale, clipping range and Apply them into a transformer block."""
        # pylint: disable=too-many-locals

        inpsected_mod_types = cast(AWQHook, self.hook).get_inspect_module_types(block)
        input_feat, intra_kwargs = collect_inps(
            block_inps,
            tuple([*self.hook.get_linear_layer_types(), *inpsected_mod_types]),
            block,
            block_kwargs,
        )
        awq_args = cast(AWQConfig, self.quant_config).awq_args

        for prev_ops, linear_layers, layer_name, module2inspect, kwargs in cast(
            AWQHook, self.hook
        ).iter_inspect_modules(block, intra_kwargs):
            module_inp = input_feat[layer_name]
            scales = search_module_scale(
                module2inspect,
                linear_layers,
                module_inp,
                awq_args.quant_group_size,
                awq_args.quant_bit,
                module_kwargs=kwargs,
            )
            scaled_module_inp = apply_module_scale(
                prev_ops,
                linear_layers,
                scales.to(self.quant_config.device),
                module_inp.to(self.quant_config.device),
            )

            if any(
                (
                    avoid in layer_name
                    for avoid in cast(AWQHook, self.hook).avoid_clipping_layer_names
                )
            ):
                continue
            max_val = search_module_clip(
                linear_layers,
                scaled_module_inp,
                awq_args.quant_group_size,
                awq_args.quant_bit,
                n_sample_token=self.quant_config.calibration_dataset.num_samples,
            )
            apply_module_clip(
                max_val.to(self.quant_config.device),
                linear_layers,
            )

        del input_feat

    @torch.no_grad()
    def _apply_awq_scale_clip(
        self,
        model: torch.nn.Module,
    ) -> None:
        """Search AWQ scale, clipping range and Apply them into model."""
        # pylint: disable=too-many-locals
        model.eval()
        model.to(self.quant_config.device)
        tf_blocks = self.hook.get_tf_blocks(model)
        block_inps, block_kwargs = collect_inps(
            inps=self.get_batched_samples(),
            target_classes=(type(tf_blocks[0]),),
            module=model,
            module_kwargs={},
        )
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        for block, inps, kwargs in tqdm(
            zip(
                tf_blocks,
                block_inps.values(),
                block_kwargs.values(),
            ),
            total=len(tf_blocks),
            desc="Search and Apply AWQ Scale, Clip range..",
        ):
            block = block.to(self.quant_config.device)
            self._apply_awq_scale_clip_block(block, inps, kwargs)
            block.to("cpu")
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
    ) -> Iterator[TFQuantResults]:
        """Quantize model with SmoothQuant."""
        model.eval()
        model.to(self.quant_config.device)
        for quant_input in tqdm(
            self.hook.iter_quant_inputs(model),
            total=len(self.hook.get_tf_blocks(model)),
            desc="Quantize model..",
        ):
            yield cast(AWQHook, self.hook).get_quant_result(
                quant_input, quant_config=cast(AWQConfig, self.quant_config)
            )

    def get_quantized_state_dict(
        self, model: torch.nn.Module, quant_result_iter: Iterator[TFQuantResults]
    ) -> Dict[str, torch.Tensor]:
        """Get quantized state dict for AWQ."""
        state_dict = model.state_dict()
        for quant_result in quant_result_iter:
            for field in fields(quant_result):
                layer_name = field.name
                layer = getattr(quant_result, layer_name)

                if isinstance(layer, WeightOnlyQuantResult):
                    state_dict[
                        f"{quant_result.layer_prefix_with_index}{layer_name}.woq_weight"
                    ] = layer.q_weight
                    state_dict[
                        f"{quant_result.layer_prefix_with_index}{layer_name}.woq_weight_scale"
                    ] = layer.weight_scale
                    state_dict[
                        f"{quant_result.layer_prefix_with_index}{layer_name}.woq_weight_zp"
                    ] = layer.zero_point

        return state_dict
