# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Int8 Quantizer Base."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Tuple, cast

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel  # type: ignore

from friendli.modules.converter.utils import get_tokenizer
from friendli.modules.quantizer_v2.base import (
    AbstractQuantHookV2,
    AbstractQuantizerV2,
    AbstractWeightActQuantizer,
    AbstractWeightOnlyQuantizer,
)
from friendli.modules.quantizer_v2.int8.utils import perform_smoothing
from friendli.modules.quantizer_v2.schema.config import Int8QuantConfig
from friendli.modules.quantizer_v2.schema.data import ModuleName
from friendli.modules.quantizer_v2.utils import collect_stats, safe_load_datasets


class Int8QuantHook(AbstractQuantHookV2):
    """Int8 Quant Hook Base."""

    @abstractmethod
    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the attention fc layer in the decoder block."""

    @abstractmethod
    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the second feed-forward layer in the decoder block."""

    @abstractmethod
    def iter_pre_act_post_act_params(
        self, model: PreTrainedModel
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of pre_act_params and post_act_params per transformer block."""


class Int8Quantizer(AbstractQuantizerV2):
    """Int8 Quantizer Base."""

    def get_smoothing_calib_dataloader(self) -> DataLoader:
        """Get calibration dataset for Int8."""
        data_cfg = self.config.calibration_dataset
        dataset = safe_load_datasets(data_cfg)
        tokenizer = get_tokenizer(self.hook.model_config.name_or_path)
        dataset = (
            dataset.shuffle(self.config.seed)
            .select(range(data_cfg.num_samples))
            .select_columns([data_cfg.lookup_column_name])
        )
        encoded_dataset = tokenizer(
            dataset[data_cfg.lookup_column_name],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=data_cfg.max_length,
        )
        return DataLoader(encoded_dataset["input_ids"], batch_size=data_cfg.batch_size)

    def _smooth(
        self,
        model: PreTrainedModel,
    ) -> None:
        """Smooths the models before Quantization."""
        model.eval()
        # collect stats for Int8 quantization scale.
        with self._try_offload_model(model):
            calib_dataloader = self.get_smoothing_calib_dataloader()
            quant_config = cast(Int8QuantConfig, self.config)
            max_input_stats, _ = collect_stats(
                model,
                quant_config.device,
                calib_dataloader,
                self.hook.get_linear_layer_types(),
                tqdm_desc="Collecting stats for Smoothing.",
                percentile=100.0,
            )

            for pre_act_params, post_act_params, name in cast(
                Int8QuantHook, self.hook
            ).iter_pre_act_post_act_params(model):
                perform_smoothing(
                    pre_act_params,
                    post_act_params,
                    max_input_stats[name],
                    migration_strength=quant_config.int8_args.migration_strength,
                    inplace=True,
                )

    def pre_quantize(
        self,
        model: PreTrainedModel,
    ) -> None:
        """Pre-procedure that should be called before quantize() is called."""
        self._smooth(model)

    def quantize(self, model: PreTrainedModel) -> torch.nn.Module:
        """Quantize the model."""
        self.pre_quantize(model)
        return super().quantize(model)

    def get_quant_config(self) -> Dict[str, Any]:
        """Get the quantization configuration."""
        return {
            "bits": 8,
            "mode": cast(Int8QuantConfig, self.config).int8_args.quant_type.value,
            "zero_point": False,
            "quant_method": "int8",
            "quant_group_size": self.config.quant_group_size,
        }


class Int8StaticQuantizer(Int8Quantizer, AbstractWeightActQuantizer):
    """Int8 Dynamic Quantizer Base."""


class Int8DynamicQuantizer(Int8Quantizer, AbstractWeightOnlyQuantizer):
    """Int8 Dynamic Quantizer Base."""
