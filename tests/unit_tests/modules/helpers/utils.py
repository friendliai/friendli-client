# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
from accelerate import init_empty_weights
from pydantic import BaseModel
from transformers import PretrainedConfig

from periflow.enums import CheckpointDataType
from periflow.modules.converter.maps import get_hf_converter_factory
from periflow.modules.converter.utils import get_model_arch

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


def get_numpy_data_type(data_type: CheckpointDataType) -> np.dtype:
    if data_type == CheckpointDataType.FP32:
        return np.float32
    elif data_type == CheckpointDataType.FP16:
        return np.float16
    elif data_type == CheckpointDataType.BF16:
        return np.uint32
    else:
        return np.int8


def get_param_specs(model_name: str, model_config: ModelConfig) -> Dict[str, ParamInfo]:
    file_path = f"{SPEC_PATH_PREFIX}models/{model_name}.yaml"
    template = Template.from_file(file_path)
    rendered = template.render(**model_config.model_dump())
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
