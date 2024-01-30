# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Dict

import pytest
from peft import LoraConfig

from friendli.modules.converter.base import OneOfConverter
from friendli.modules.converter.maps import get_adapter_converter_factory
from friendli.modules.converter.utils import get_model_arch, get_tensor_from_state_dict

from tests.unit_tests.modules.conftest import model_name_config_map
from tests.unit_tests.modules.helpers.utils import (
    LoraAdapterConfig,
    get_meta_model_with_adapter,
    get_numpy_data_type,
    get_param_specs,
)

model_with_adpater = ["mpt"]
model_with_adpater_name_config_map = {}
for model_name, model_config in model_name_config_map.items():
    if model_name in model_with_adpater:
        model_with_adpater_name_config_map[model_name] = model_config


@pytest.fixture
def adapter_config() -> LoraConfig:
    return LoraConfig()


@pytest.fixture
def render_lora_adapter_config(
    converter: OneOfConverter, adapter_config: LoraConfig
) -> LoraAdapterConfig:
    return LoraAdapterConfig(
        dtype="float16",
        num_decoder_layers=converter.decoder_layer_num,
        hidden_size=converter.decoder_hidden_size,
        num_heads=converter.decoder_num_attention_heads,
        num_kv_heads=converter.decoder_num_kv_attention_heads,
        head_size=converter.decoder_head_size,
        num_encoder_layers=converter.decoder_layer_num,  # same as decoder for test
        ff_intermediate_size=converter.decoder_ff_intermediate_size,
        lora_rank_dim=adapter_config.r,
    )


@pytest.fixture
def lora_spec_data(
    model_name: str, render_lora_adapter_config: LoraAdapterConfig
) -> Dict[str, Any]:
    param_specs = get_param_specs(model_name, "lora", render_lora_adapter_config)
    return param_specs


@pytest.mark.parametrize(
    "model_config",
    model_with_adpater_name_config_map.values(),
)
def test_convert_info_list_match_hf_state_dict(
    converter: OneOfConverter,
    adapter_config: LoraConfig,
):
    model_arch = get_model_arch(converter.config)
    adapter_converter_cls = get_adapter_converter_factory(model_arch)
    adapter_converter = adapter_converter_cls(converter, adapter_config)

    convert_info_list = adapter_converter.get_convert_info_list()
    model_with_adapter = get_meta_model_with_adapter(
        adapter_converter.converter.config, adapter_converter.adapter_config
    )
    state_dict = model_with_adapter.state_dict()
    for convert_info in convert_info_list:
        param_names = convert_info.param_names
        for param_name in param_names:
            assert param_name in state_dict


@pytest.mark.parametrize(
    "model_name, model_config",
    model_with_adpater_name_config_map.items(),
)
def test_model_with_lora_match_spec(
    converter: OneOfConverter,
    lora_spec_data: Dict[str, Any],
    adapter_config: LoraConfig,
):
    model_arch = get_model_arch(converter.config)
    adapter_converter_cls = get_adapter_converter_factory(model_arch)
    adapter_converter = adapter_converter_cls(converter, adapter_config)

    convert_info_list = adapter_converter.get_convert_info_list()
    model_with_adapter = get_meta_model_with_adapter(
        adapter_converter.converter.config, adapter_converter.adapter_config
    )
    state_dict = model_with_adapter.state_dict()
    for convert_info in convert_info_list:
        converted_name, reshape_fn, param_names, data_type = (
            convert_info.converted_name,
            convert_info.reshape_fn,
            convert_info.param_names,
            convert_info.data_type,
        )
        assert lora_spec_data[converted_name].dtype == get_numpy_data_type(
            data_type
        ), f"data type mismatch for {converted_name}: {param_names}"
        params = [
            get_tensor_from_state_dict(state_dict, param_name)
            for param_name in param_names
        ]
        reshaped_tensor = reshape_fn(params)
        assert (
            lora_spec_data[converted_name].shape == reshaped_tensor.shape
        ), f"shape mismatch for {converted_name}: {param_names}"
