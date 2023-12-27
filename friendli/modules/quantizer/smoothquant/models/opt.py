# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli OPTForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Iterator, List, Tuple, Type, cast

import torch
from transformers.models.opt import OPTForCausalLM  # type: ignore[import]

from friendli.modules.quantizer.schema.config import SmoothQuantConfig
from friendli.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from friendli.modules.quantizer.smoothquant.base import SmoothQuantHook


class SmoothQuantOPTHook(SmoothQuantHook):
    """SmoothQuant Hook for OPTForCausalLM."""

    def iter_smooth_norm_weights(
        self, model: OPTForCausalLM
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in OPTForCausalLM."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args
        for index, decoder_layer in enumerate(model.model.decoder.layers):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection ] gets smoothed
            yield (
                [
                    decoder_layer.self_attn_layer_norm.weight.data,
                    decoder_layer.self_attn_layer_norm.bias.data,
                ],
                [
                    decoder_layer.self_attn.q_proj.weight.data,
                    decoder_layer.self_attn.k_proj.weight.data,
                    decoder_layer.self_attn.v_proj.weight.data,
                ],
                f"{self.quantized_layer_prefix}{index}.self_attn.q_proj",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.final_layer_norm.weight.data,
                    decoder_layer.final_layer_norm.bias.data,
                ],
                [decoder_layer.fc1.weight.data],
                f"{self.quantized_layer_prefix}{index}.fc1",
            )
            if quant_args.attn_fc_smoothing:
                yield (
                    [decoder_layer.attn_fc_pre_smoother.scale.data],
                    [decoder_layer.self_attn.out_proj.weight.data],
                    f"{self.quantized_layer_prefix}{index}.self_attn.out_proj",
                )
            if quant_args.ff2_smoothing:
                yield (
                    [decoder_layer.ff2_pre_smoother.scale.data],
                    [decoder_layer.fc2.weight.data],
                    f"{self.quantized_layer_prefix}{index}.fc2",
                )

    def iter_tf_quant_inputs(self, model: OPTForCausalLM) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of OPTForCausalLM."""
        for index, decoder_layer in enumerate(model.model.decoder.layers):
            self_attn = decoder_layer.self_attn
            fc1 = decoder_layer.fc1
            fc2 = decoder_layer.fc2
            yield TFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                q=QuantInput(
                    self_attn.q_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.self_attn.q_proj",
                    None,
                    None,
                ),
                k=QuantInput(
                    self_attn.k_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.self_attn.k_proj",
                    None,
                    None,
                ),
                v=QuantInput(
                    self_attn.v_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.self_attn.v_proj",
                    None,
                    None,
                ),
                attn_fc=QuantInput(
                    self_attn.out_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.self_attn.out_proj",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}{index}.fc1",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}{index}.fc2",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in OPTForCausalLM."""
        return (torch.nn.Linear,)

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after attention in the decoder layer."""
        return decoder_layer.self_attn.out_proj

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after FF1 in the decoder layer."""
        return decoder_layer.fc2

    def get_tf_blocks(self, model: OPTForCausalLM) -> List[torch.nn.Module]:
        """Returns the decoder layers(transformer blocks) in the model."""
        return list(model.model.decoder.layers)
