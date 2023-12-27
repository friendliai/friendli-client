# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Dict

import pytest

from friendli.modules.converter.base import OneOfConverter
from friendli.modules.converter.utils import get_tensor_from_state_dict

from tests.unit_tests.modules.conftest import model_name_config_map
from tests.unit_tests.modules.helpers.utils import get_meta_model, get_numpy_data_type


@pytest.mark.parametrize(
    "model_config",
    model_name_config_map.values(),
)
def test_convert_info_list_match_hf_state_dict(converter: OneOfConverter):
    convert_info_list = converter.get_convert_info_list()
    assert len(convert_info_list) != 0
    model = get_meta_model(converter.config)
    state_dict = model.state_dict()
    for convert_info in convert_info_list:
        param_names = convert_info.param_names
        for param_name in param_names:
            assert param_name in state_dict


@pytest.mark.parametrize(
    "model_name, model_config",
    model_name_config_map.items(),
)
def test_convert_info_list_match_spec(
    converter: OneOfConverter, spec_data: Dict[str, Any]
):
    convert_info_list = converter.get_convert_info_list()
    assert len(convert_info_list) != 0
    converted_param_names = set()
    for convert_info in convert_info_list:
        converted_param_names.add(convert_info.converted_name)

    spec_converted_param_names = set(spec_data.keys())
    assert converted_param_names == spec_converted_param_names


@pytest.mark.parametrize(
    "model_name, model_config",
    model_name_config_map.items(),
)
def test_reshape_fn_match_spec(converter: OneOfConverter, spec_data: Dict[str, Any]):
    convert_info_list = converter.get_convert_info_list()
    model = get_meta_model(converter.config)
    state_dict = model.state_dict()
    for convert_info in convert_info_list:
        converted_name, reshape_fn, param_names, data_type = (
            convert_info.converted_name,
            convert_info.reshape_fn,
            convert_info.param_names,
            convert_info.data_type,
        )
        assert spec_data[converted_name].dtype == get_numpy_data_type(
            data_type
        ), f"data type mismatch for {converted_name}: {param_names}"
        params = [
            get_tensor_from_state_dict(state_dict, param_name)
            for param_name in param_names
        ]
        reshaped_tensor = reshape_fn(params)
        assert (
            spec_data[converted_name].shape == reshaped_tensor.shape
        ), f"shape mismatch for {converted_name}: {param_names}"
