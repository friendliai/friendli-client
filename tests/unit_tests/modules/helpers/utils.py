# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

import os
from dataclasses import fields
from typing import Dict, Optional
from unittest.mock import Mock

import numpy as np
import torch
from accelerate import init_empty_weights
from peft import PeftConfig, PeftModel
from pydantic import BaseModel
from transformers import PretrainedConfig

from friendli.enums import ModelDataType
from friendli.modules.converter.maps import (
    get_adapter_converter_factory,
    get_hf_converter_factory,
)
from friendli.modules.converter.utils import get_model_arch
from friendli.modules.quantizer.awq.base import AWQQuantizer
from friendli.modules.quantizer.layers import (
    WeightActQuantizedLinearLayer,
    WeightOnlyQuantizedLinearLayer,
)
from friendli.modules.quantizer.schema.config import AWQConfig
from friendli.modules.quantizer.schema.data import QuantInput
from friendli.modules.quantizer.smoothquant.base import SmoothQuantQuantizer
from friendli.utils.compat import model_dump

from tests.unit_tests.modules.helpers.spec import ModelSpecParser, ParamInfo, Template

SPEC_PATH_PREFIX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "specs/"
)


class ModelConfig(BaseModel):
    """Adjustable model config."""

    dtype: str
    num_decoder_layers: int
    hidden_size: int
    num_encoder_layers: Optional[int] = None
    ff_intermediate_size: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_size: Optional[int] = None
    seq_len: Optional[int] = 1024
    vocab_size: Optional[int] = 10000
    num_experts: Optional[int] = 8


class LoraAdapterConfig(ModelConfig):
    """Adjustable model config."""

    lora_rank_dim: int


class AWQModelConfig(ModelConfig):
    """Adjustable model config for AWQ."""

    group_size: int = 1
    q_dtype: str = "int8"


class SmoothQuantModelConfig(ModelConfig):
    """Adjustable model config for SmoothQuant."""

    attn_fc_smoothing: bool = False
    ff2_smoothing: bool = False
    q_dtype: str = "int8"


def get_numpy_data_type(data_type: ModelDataType) -> np.dtype:
    if data_type == ModelDataType.FP32:
        return np.float32
    elif data_type == ModelDataType.FP16:
        return np.float16
    elif data_type == ModelDataType.BF16:
        return np.uint32
    else:
        return np.int8


def get_param_specs(
    model_name: str, spec_folder: str, model_config: ModelConfig
) -> Dict[str, ParamInfo]:
    file_path = f"{SPEC_PATH_PREFIX}{spec_folder}/{model_name}.yaml"
    template = Template.from_file(file_path)
    render_config = model_dump(model_config)
    rendered = template.render(**render_config)
    assert isinstance(rendered, dict)
    parser = ModelSpecParser(model_spec=rendered)
    param_specs = parser.get_all_param_info()
    return param_specs


def get_meta_model(
    model_config: PretrainedConfig,
) -> torch.nn.Module:
    model_arch = get_model_arch(model_config)
    model_factory, _ = get_hf_converter_factory(model_arch)
    with init_empty_weights():
        model = model_factory(config=model_config)
    return model


def get_meta_model_with_adapter(
    model_config: PretrainedConfig, adapter_config: PeftConfig
) -> torch.nn.Module:
    model_arch = get_model_arch(model_config)
    model_factory, _ = get_hf_converter_factory(model_arch)
    with init_empty_weights():
        model = model_factory(config=model_config)
        PeftModel(model, adapter_config)
    return model


def get_smoothquant_quantized_meta_model(
    model_config: PretrainedConfig, quantizer: SmoothQuantQuantizer
):
    model = get_meta_model(model_config)
    model = quantizer.hook.pre_smooth(model).to("meta")

    def weight_act_quant_layer(quant_input: QuantInput):
        weight, start, end = (
            quant_input.weight,
            quant_input.start_offset,
            quant_input.end_offset,
        )
        weight = weight[start:end]
        return WeightActQuantizedLinearLayer(  # meta quantized linear layer
            in_features=weight.size(1),
            out_features=weight.size(0),
            q_weight=weight,
            weight_scale=torch.zeros(weight.size(1), device="meta"),
            act_scale=torch.zeros(weight.size(1), device="meta"),
        )

    for tf_quant_input in quantizer.hook.iter_tf_quant_inputs(model):
        for field in fields(tf_quant_input):
            quant_input = getattr(tf_quant_input, field.name)
            if isinstance(quant_input, QuantInput):
                weight_act_quant_layer = Mock(side_effect=weight_act_quant_layer)
                q_layer = weight_act_quant_layer(quant_input)
                tf_quant_input.block.add_module(field.name, q_layer)

    return model


def get_awq_quantized_meta_model(
    model_config: PretrainedConfig, quantizer: AWQQuantizer, quant_config: AWQConfig
):
    model = get_meta_model(model_config)
    model = quantizer.hook.add_pre_scaler(model).to("meta")

    def weight_act_quant_layer(quant_input: QuantInput):
        weight, start, end = (
            quant_input.weight,
            quant_input.start_offset,
            quant_input.end_offset,
        )
        w = weight[start:end]
        out_dim = w.size(0)
        in_dim = w.size(1)
        num_groups = in_dim // quant_config.awq_args.quant_group_size
        return WeightOnlyQuantizedLinearLayer(  # meta quantized linear layer
            in_features=in_dim,
            out_features=out_dim,
            q_weight=w,
            weight_scale=torch.zeros((num_groups, out_dim), device="meta"),
            zeros=torch.zeros((num_groups, out_dim), device="meta"),
        )

    for tf_quant_input in quantizer.hook.iter_tf_quant_inputs(model):
        for field in fields(tf_quant_input):
            quant_input = getattr(tf_quant_input, field.name)
            if isinstance(quant_input, QuantInput):
                weight_only_quantzer = Mock(side_effect=weight_act_quant_layer)
                q_layer = weight_only_quantzer(quant_input)
                tf_quant_input.block.add_module(field.name, q_layer)

    return model
