# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# # Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# from __future__ import annotations

# from typing import Any, Dict

# import pytest

# from friendli.modules.converter.base import OneOfConverter
# from friendli.modules.converter.utils import get_tensor_from_state_dict
# from friendli.modules.quantizer.maps import get_quantized_converter
# from friendli.modules.quantizer.schema.config import SmoothQuantArgs, SmoothQuantConfig

# from tests.unit_tests.modules.conftest import model_name_config_map
# from tests.unit_tests.modules.helpers.utils import (
#     SmoothQuantModelConfig,
#     get_numpy_data_type,
#     get_param_specs,
#     get_smoothquant_quantized_meta_model,
# )

# smoothquant_models = [
#     "bloom",
#     "codegen",
#     "falcon",
#     "falcon_7b",
#     "gpt_j",
#     "gpt_neox",
#     "llama",
#     "mpt",
#     "opt",
# ]
# smoothquant_model_name_config_map = {}
# for model_name, model_config in model_name_config_map.items():
#     if model_name in smoothquant_models:
#         smoothquant_model_name_config_map[model_name] = model_config


# @pytest.fixture
# def quant_config() -> SmoothQuantConfig:
#     return SmoothQuantConfig(
#         smoothquant_args=SmoothQuantArgs(
#             attn_fc_smoothing=True,
#             ff2_smoothing=True,
#         )
#     )


# @pytest.fixture
# def render_smoothquant_model_config(
#     converter: OneOfConverter, quant_config: SmoothQuantConfig
# ) -> SmoothQuantModelConfig:
#     return SmoothQuantModelConfig(
#         dtype="float16",
#         num_decoder_layers=converter.decoder_layer_num,
#         hidden_size=converter.decoder_hidden_size,
#         num_heads=converter.decoder_num_attention_heads,
#         num_kv_heads=converter.decoder_num_kv_attention_heads,
#         head_size=converter.decoder_head_size,
#         num_encoder_layers=converter.decoder_layer_num,  # same as decoder for test
#         ff_intermediate_size=converter.decoder_ff_intermediate_size,
#         attn_fc_smoothing=quant_config.smoothquant_args.attn_fc_smoothing,
#         ff2_smoothing=quant_config.smoothquant_args.ff2_smoothing,
#         q_dtype="int8",
#     )


# @pytest.fixture
# def smoothquant_spec_data(
#     model_name: str, render_smoothquant_model_config: SmoothQuantModelConfig
# ) -> Dict[str, Any]:
#     param_specs = get_param_specs(
#         model_name, "smoothquant", render_smoothquant_model_config
#     )
#     return param_specs


# @pytest.mark.parametrize(
#     "model_config",
#     smoothquant_model_name_config_map.values(),
# )
# def test_convert_info_list_match_hf_state_dict(
#     converter: OneOfConverter, quant_config: SmoothQuantConfig
# ):
#     quantizer = get_quantized_converter(quant_config, converter)
#     convert_info_list = quantizer.get_convert_info_list()
#     assert len(convert_info_list) != 0
#     quantized_model = get_smoothquant_quantized_meta_model(converter.config, quantizer)
#     state_dict = quantized_model.state_dict()
#     for convert_info in convert_info_list:
#         param_names = convert_info.param_names
#         for param_name in param_names:
#             assert param_name in state_dict


# @pytest.mark.parametrize(
#     "model_name, model_config",
#     smoothquant_model_name_config_map.items(),
# )
# def test_quantized_model_match_spec(
#     converter: OneOfConverter,
#     smoothquant_spec_data: Dict[str, Any],
#     quant_config: SmoothQuantConfig,
# ):
#     quantizer = get_quantized_converter(quant_config, converter)
#     quantized_model = get_smoothquant_quantized_meta_model(converter.config, quantizer)
#     state_dict = quantized_model.state_dict()
#     convert_info_list = quantizer.get_convert_info_list()
#     for convert_info in convert_info_list:
#         converted_name, reshape_fn, param_names, data_type = (
#             convert_info.converted_name,
#             convert_info.reshape_fn,
#             convert_info.param_names,
#             convert_info.data_type,
#         )
#         assert smoothquant_spec_data[converted_name].dtype == get_numpy_data_type(
#             data_type
#         ), f"data type mismatch for {converted_name}: {param_names}"
#         params = [
#             get_tensor_from_state_dict(state_dict, param_name)
#             for param_name in param_names
#         ]
#         reshaped_tensor = reshape_fn(params)
#         assert (
#             smoothquant_spec_data[converted_name].shape == reshaped_tensor.shape
#         ), f"shape mismatch for {converted_name}: {param_names}"
