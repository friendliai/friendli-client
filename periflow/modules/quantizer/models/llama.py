# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow LlamaForCausalLM QuantizerHook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, cast

import torch
from transformers.models.llama import LlamaForCausalLM  # type: ignore[import]

from periflow.modules.quantizer.base import (
    Int8QuantScaleInput,
    Int8QuantScaleInputTuple,
    SmoothQuantHook,
)
from periflow.modules.quantizer.formatter import (
    Int8QuantScale,
    Int8QuantScaleInput,
    Int8QuantScaleResult,
    ModuleName,
)
from periflow.modules.quantizer.utils import get_int8_quant_scales


@dataclass
class LlamaInt8QuantScaleInput(Int8QuantScaleInput):
    """Dataclass for int8 quantization input per layer in LlamaForCausalLM.""" ""

    gate_ff: Int8QuantScaleInputTuple


@dataclass
class LlamaInt8QuantScaleResult(Int8QuantScaleResult):
    """Dataclass for int8 quantization result per a transformer layer in LlamaForCausalLM.""" ""

    gate_ff: Int8QuantScale


class SmoothQuantLlamaHook(SmoothQuantHook):
    """SmoothQuant Hook for LlamaForCausalLM."""

    def get_smooth_norm_weights(self, model: LlamaForCausalLM):
        """Returns iterator of layernorm's weight and linear layer's weight per transformer layer in LlamaForCausalLM."""
        for index, decoder_layer in enumerate(model.model.layers):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection ] gets smoothed
            yield (
                [
                    decoder_layer.input_layernorm.weight.data,
                ],
                [
                    decoder_layer.self_attn.q_proj.weight.data,
                    decoder_layer.self_attn.k_proj.weight.data,
                    decoder_layer.self_attn.v_proj.weight.data,
                ],
                f"{self.quantized_layer_prefix}.{index}.self_attn.q_proj",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.post_attention_layernorm.weight.data,
                ],
                [
                    decoder_layer.mlp.up_proj.weight.data,
                    decoder_layer.mlp.gate_proj.weight.data,
                ],
                f"{self.quantized_layer_prefix}.{index}.mlp.up_proj",  # the input tensors fed into up_proj, gate_proj matrices are identical.
            )

    def get_quant_inputs(self, model: LlamaForCausalLM):
        """Returns the layers which should be quantized in transformer layer of LlamaForCausalLM."""
        for index, decoder_layer in enumerate(model.model.layers):
            self_attn = decoder_layer.self_attn
            fc1 = decoder_layer.mlp.up_proj
            gate_ff = decoder_layer.mlp.gate_proj
            fc2 = decoder_layer.mlp.down_proj

            yield LlamaInt8QuantScaleInput(
                layer_index=index,
                q=(
                    self_attn.q_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.q_proj",
                    None,
                    None,
                ),
                k=(
                    self_attn.k_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.k_proj",
                    None,
                    None,
                ),
                v=(
                    self_attn.v_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.v_proj",
                    None,
                    None,
                ),
                attn_fc=(
                    self_attn.o_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.o_proj",
                    None,
                    None,
                ),
                ff1=(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.up_proj",
                    None,
                    None,
                ),
                gate_ff=(
                    gate_ff.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.gate_proj",
                    None,
                    None,
                ),
                ff2=(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.down_proj",
                    None,
                    None,
                ),
            )

    @staticmethod
    def get_quant_result(
        quant_input: Int8QuantScaleInput,
        max_input_stats: Dict[ModuleName, torch.Tensor],
        max_output_stats: Dict[ModuleName, torch.Tensor],
    ) -> Int8QuantScaleResult:
        """Returns the quantization result for a specific layer in LlamaForCausalLM."""

        def get_scale(f: Int8QuantScaleInputTuple) -> Int8QuantScale:
            weight, name, start, end = f
            return get_int8_quant_scales(
                name,
                max_input_stats[name],
                weight[start:end],
                max_output_stats[name][start:end],
            )

        quant_input = cast(LlamaInt8QuantScaleInput, quant_input)
        return LlamaInt8QuantScaleResult(
            layer_index=quant_input.layer_index,
            q=get_scale(quant_input.q),
            k=get_scale(quant_input.k),
            v=get_scale(quant_input.v),
            attn_fc=get_scale(quant_input.attn_fc),
            ff1=get_scale(quant_input.ff1),
            gate_ff=get_scale(quant_input.gate_ff),
            ff2=get_scale(quant_input.ff2),
        )

    def get_linear_layer_types(self):
        """Returns the linear layer types in LlamaForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self):
        """Returns the prefix of the transformer layer in LlamaForCausalLM."""
        return "model.layers"
