# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli LlamaForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple, Type, cast

import torch
from transformers.models.llama import (  # type: ignore[import]
    LlamaConfig,
    LlamaForCausalLM,
)

from friendli.modules.converter.base import DECODER_PREFIX, OneOfConverter
from friendli.modules.quantizer.schema.config import SmoothQuantConfig
from friendli.modules.quantizer.schema.data import (
    ModuleName,
    QuantInput,
    TFQuantInputs,
    TFQuantResults,
    WeightActQuantResult,
)
from friendli.modules.quantizer.smoothquant.base import SmoothQuantHook
from friendli.modules.quantizer.utils import get_weight_act_quant_scales


@dataclass
class LlamaTFQuantInput(TFQuantInputs):
    """Dataclass for int8 quantization input per layer in LlamaForCausalLM.""" ""

    ff_gate: QuantInput


@dataclass
class LlamaTFQuantResults(TFQuantResults):
    """Dataclass for int8 quantization result per a transformer block in LlamaForCausalLM.""" ""

    ff_gate: WeightActQuantResult


class SmoothQuantLlamaHook(SmoothQuantHook):
    """SmoothQuant Hook for LlamaForCausalLM."""

    def __init__(self, quant_config: SmoothQuantConfig, converter: OneOfConverter):
        """Initialize SmoothQuantLlamaHook."""
        super().__init__(quant_config, converter)
        config = cast(LlamaConfig, converter.config)
        self.num_attention_heads = config.num_attention_heads
        if config.num_key_value_heads is None:
            self.num_kv_attention_heads = self.num_attention_heads
        else:
            self.num_kv_attention_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_dim = self.head_size

    def pre_smooth(self, model: torch.nn.Module) -> torch.nn.Module:
        """Pre-procedure for SmoothQuant in LlamaForCausalLM that should be called before smooth() is called."""
        super().pre_smooth(model)
        for decoder_layer in cast(LlamaForCausalLM, model).model.layers:
            decoder_layer.add_module(
                "post_attention_layernorm_2",
                copy.deepcopy(decoder_layer.post_attention_layernorm),
            )
        return model

    def iter_smooth_norm_weights(
        self,
        model: LlamaForCausalLM,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in LlamaForCausalLM."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args

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
                f"{self.quantized_layer_prefix}{index}.self_attn.q_proj",  # the input tensors fed into Q, K, V matrices are identical.
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.post_attention_layernorm.weight.data,
                ],
                [
                    decoder_layer.mlp.up_proj.weight.data,
                ],
                f"{self.quantized_layer_prefix}{index}.mlp.up_proj",
            )
            # [LayerNomr 2] = [ MLP GATED FF ] gets smoothed
            yield (
                [
                    decoder_layer.post_attention_layernorm_2.weight.data,
                ],
                [
                    decoder_layer.mlp.gate_proj.weight.data,
                ],
                f"{self.quantized_layer_prefix}{index}.mlp.gate_proj",
            )
            if quant_args.attn_fc_smoothing:
                yield (
                    [decoder_layer.attn_fc_pre_smoother.scale.data],
                    [decoder_layer.self_attn.o_proj.weight.data],
                    f"{self.quantized_layer_prefix}{index}.self_attn.o_proj",
                )

            if quant_args.ff2_smoothing:
                yield (
                    [decoder_layer.ff2_pre_smoother.scale.data],
                    [decoder_layer.mlp.down_proj.weight.data],
                    f"{self.quantized_layer_prefix}{index}.mlp.down_proj",
                )

    def iter_tf_quant_inputs(self, model: LlamaForCausalLM) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of LlamaForCausalLM."""
        for index, decoder_layer in enumerate(model.model.layers):
            self_attn = decoder_layer.self_attn
            fc1 = decoder_layer.mlp.up_proj
            ff_gate = decoder_layer.mlp.gate_proj
            fc2 = decoder_layer.mlp.down_proj

            yield LlamaTFQuantInput(
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
                    self_attn.o_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.self_attn.o_proj",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.up_proj",
                    None,
                    None,
                ),
                ff_gate=QuantInput(
                    ff_gate.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.gate_proj",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.down_proj",
                    None,
                    None,
                ),
            )

    def get_quant_result(
        self,
        quant_input: TFQuantInputs,
        **kwargs: Any,
    ) -> TFQuantResults:
        """Returns the quantization result for a specific layer in LlamaForCausalLM."""
        max_input_stats: Dict[ModuleName, torch.Tensor] = kwargs["max_input_stats"]
        max_output_stats: Dict[ModuleName, torch.Tensor] = kwargs["max_output_stats"]

        def get_scale(quant_input: QuantInput) -> WeightActQuantResult:
            weight, name, start, end = (
                quant_input.weight,
                quant_input.name,
                quant_input.start_offset,
                quant_input.end_offset,
            )
            return get_weight_act_quant_scales(
                name,
                max_input_stats[name],
                weight[start:end],
                max_output_stats[name][start:end],
            )

        quant_input = cast(LlamaTFQuantInput, quant_input)
        return LlamaTFQuantResults(
            layer_prefix_with_index=f"{self.quantized_layer_prefix}{quant_input.layer_index}.",
            q=get_scale(quant_input.q),
            k=get_scale(quant_input.k),
            v=get_scale(quant_input.v),
            attn_fc=get_scale(quant_input.attn_fc),
            ff1=get_scale(quant_input.ff1),
            ff_gate=get_scale(quant_input.ff_gate),
            ff2=get_scale(quant_input.ff2),
        )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in LlamaForCausalLM."""
        return (torch.nn.Linear,)

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after attention in the decoder layer."""
        return decoder_layer.self_attn.o_proj

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after FF1 in the decoder layer."""
        return decoder_layer.mlp.down_proj

    def get_tf_blocks(self, model: LlamaForCausalLM) -> List[torch.nn.Module]:
        """Returns the decoder layers(transformer blocks) in the model."""
        return list(model.model.layers)

    @property
    def quantized_param_names(self) -> List[str]:
        """Returns the parameter names in LlamaForCausalLM."""
        param_names = super().quantized_param_names
        for i in range(self.converter.decoder_layer_num):
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            param_names.append(f"{converted_prefix}mlp/c_gate/weight:0")
        return param_names
