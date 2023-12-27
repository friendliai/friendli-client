# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli GPTJForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

import copy
from typing import Iterator, List, Tuple, Type, cast

import torch
from transformers.models.gptj import GPTJForCausalLM  # type: ignore[import]

from friendli.modules.converter.base import DECODER_PREFIX
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.quantizer.schema.config import SmoothQuantConfig
from friendli.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from friendli.modules.quantizer.smoothquant.base import SmoothQuantHook


class SmoothQuantGPTJHook(SmoothQuantHook):
    """SmoothQuant Hook for GPTJForCausalLM."""

    def pre_smooth(self, model: torch.nn.Module) -> torch.nn.Module:
        """Pre-procedure for SmoothQuant in GPTJForCausalLM that should be called before smooth() is called."""
        super().pre_smooth(model)
        for decoder_layer in cast(GPTJForCausalLM, model).transformer.h:
            decoder_layer.add_module("ln_2", copy.deepcopy(decoder_layer.ln_1))
        return model

    def iter_smooth_norm_weights(
        self,
        model: GPTJForCausalLM,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in GPTJForCausalLM."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args
        for index, decoder_layer in enumerate(model.transformer.h):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection] gets smoothed
            yield (
                [
                    decoder_layer.ln_1.weight.data,
                    decoder_layer.ln_1.bias.data,
                ],
                [
                    decoder_layer.attn.q_proj.weight.data,  # [OutDim, InDim]
                    decoder_layer.attn.k_proj.weight.data,  # [OutDim, InDim]
                    decoder_layer.attn.v_proj.weight.data,  # [OutDim, InDim]
                ],
                f"{self.quantized_layer_prefix}{index}.attn.q_proj",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 1] - [ MLP FF1 ] gets smoothed
            yield (
                [
                    decoder_layer.ln_2.weight.data,
                    decoder_layer.ln_2.bias.data,
                ],
                [
                    decoder_layer.mlp.fc_in.weight.data,
                ],
                f"{self.quantized_layer_prefix}{index}.mlp.fc_in",
            )
            if quant_args.attn_fc_smoothing:
                yield (
                    [decoder_layer.attn_fc_pre_smoother.scale.data],
                    [decoder_layer.attn.out_proj.weight.data],
                    f"{self.quantized_layer_prefix}{index}.attn.out_proj",
                )
            if quant_args.ff2_smoothing:
                yield (
                    [decoder_layer.ff2_pre_smoother.scale.data],
                    [decoder_layer.mlp.fc_out.weight.data],
                    f"{self.quantized_layer_prefix}{index}.mlp.fc_out",
                )

    def iter_tf_quant_inputs(self, model: GPTJForCausalLM) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of GPTJForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            attn = decoder_layer.attn
            fc1 = decoder_layer.mlp.fc_in
            fc2 = decoder_layer.mlp.fc_out
            yield TFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                q=QuantInput(
                    attn.q_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.q_proj",
                    None,
                    None,
                ),
                k=QuantInput(
                    attn.k_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.k_proj",
                    None,
                    None,
                ),
                v=QuantInput(
                    attn.v_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.v_proj",
                    None,
                    None,
                ),
                attn_fc=QuantInput(
                    attn.out_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.out_proj",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.fc_in",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.fc_out",
                    None,
                    None,
                ),
            )

    @property
    def modified_layers_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Returns the modified layers' convert dict in GPTJForCausalLM."""
        convert_info_list = super().modified_layers_convert_info_list

        for i in range(self.converter.decoder_layer_num):
            layer_prefix = f"{self.quantized_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ln_2.weight"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}ln_2/gamma:0",
                        reshape_fn=self.converter.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ln_2.bias"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}ln_2/beta:0",
                        reshape_fn=self.converter.ln_bias_reshape,
                    ),
                ]
            )

        return convert_info_list

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in GPTJForCausalLM."""
        return (torch.nn.Linear,)

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after attention in the decoder layer."""
        return decoder_layer.attn.out_proj

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after FF1 in the decoder layer."""
        return decoder_layer.mlp.fc_out

    def get_tf_blocks(self, model: GPTJForCausalLM) -> List[torch.nn.Module]:
        """Returns the decoder layers(transformer blocks) in the model."""
        return list(model.transformer.h)
