# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Dict

import pytest

from friendli.modules.converter.base import OneOfConverter
from friendli.modules.converter.utils import get_tensor_from_state_dict
from friendli.modules.quantizer.maps import get_quantized_converter
from friendli.modules.quantizer.schema.config import AWQConfig

from tests.unit_tests.modules.conftest import model_name_config_map
from tests.unit_tests.modules.helpers.utils import (
    AWQModelConfig,
    get_awq_quantized_meta_model,
    get_numpy_data_type,
    get_param_specs,
)

awq_models = ["gptj", "gpt_neox", "llama", "mpt", "mistral"]
awq_model_name_config_map = {}
for model_name, model_config in model_name_config_map.items():
    if model_name in awq_models:
        awq_model_name_config_map[model_name] = model_config


@pytest.fixture
def quant_config() -> AWQConfig:
    return AWQConfig()


@pytest.fixture
def render_awq_model_config(
    converter: OneOfConverter, quant_config: AWQConfig
) -> AWQModelConfig:
    return AWQModelConfig(
        dtype="float16",
        num_decoder_layers=converter.decoder_layer_num,
        hidden_size=converter.decoder_hidden_size,
        num_heads=converter.decoder_num_attention_heads,
        num_kv_heads=converter.decoder_num_kv_attention_heads,
        head_size=converter.decoder_head_size,
        num_encoder_layers=converter.decoder_layer_num,  # same as decoder for test
        ff_intermediate_size=converter.decoder_ff_intermediate_size,
        group_size=quant_config.awq_args.quant_group_size,
        q_dtype="int8",
    )


@pytest.fixture
def awq_spec_data(
    model_name: str, render_awq_model_config: AWQModelConfig
) -> Dict[str, Any]:
    param_specs = get_param_specs(model_name, "awq", render_awq_model_config)
    return param_specs


@pytest.mark.parametrize(
    "model_config",
    awq_model_name_config_map.values(),
)
def test_convert_info_list_match_hf_state_dict(
    converter: OneOfConverter, quant_config: AWQConfig
):
    quantizer = get_quantized_converter(quant_config, converter)
    convert_info_list = quantizer.get_convert_info_list()
    assert len(convert_info_list) != 0
    quantized_model = get_awq_quantized_meta_model(
        converter.config, quantizer, quant_config
    )
    state_dict = quantized_model.state_dict()
    for convert_info in convert_info_list:
        param_names = convert_info.param_names
        for param_name in param_names:
            assert param_name in state_dict


@pytest.mark.parametrize(
    "model_name, model_config",
    awq_model_name_config_map.items(),
)
def test_quantized_model_match_spec(
    converter: OneOfConverter, awq_spec_data: Dict[str, Any], quant_config: AWQConfig
):
    quantizer = get_quantized_converter(quant_config, converter)
    quantized_model = get_awq_quantized_meta_model(
        converter.config, quantizer, quant_config
    )
    state_dict = quantized_model.state_dict()
    convert_info_list = quantizer.get_convert_info_list()
    for convert_info in convert_info_list:
        converted_name, reshape_fn, param_names, data_type = (
            convert_info.converted_name,
            convert_info.reshape_fn,
            convert_info.param_names,
            convert_info.data_type,
        )
        assert awq_spec_data[converted_name].dtype == get_numpy_data_type(
            data_type
        ), f"data type mismatch for {converted_name}: {param_names}"
        params = [
            get_tensor_from_state_dict(state_dict, param_name)
            for param_name in param_names
        ]
        reshaped_tensor = reshape_fn(params)
        assert (
            awq_spec_data[converted_name].shape == reshaped_tensor.shape
        ), f"shape mismatch for {converted_name}: {param_names}"
