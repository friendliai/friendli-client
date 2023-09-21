# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow LlamaForCausalLM QuantizerHook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Type, cast

import torch
from transformers import PretrainedConfig  # type: ignore[import]
from transformers.models.llama import (  # type: ignore[import]
    LlamaConfig,
    LlamaForCausalLM,
)

from periflow.modules.quantizer.base import SmoothQuantHook
from periflow.modules.quantizer.schema import (
    Int8QuantInput,
    Int8QuantResult,
    ModuleName,
    TFInt8QuantInputs,
    TFInt8QuantResults,
)
from periflow.modules.quantizer.utils import (
    convert_to_gpt_j_params,
    get_int8_quant_scales,
)


@dataclass
class LlamaTFInt8QuantInputs(TFInt8QuantInputs):
    """Dataclass for int8 quantization input per layer in LlamaForCausalLM.""" ""

    ff_gate: Int8QuantInput


@dataclass
class LlamaTFInt8QuantResults(TFInt8QuantResults):
    """Dataclass for int8 quantization result per a transformer block in LlamaForCausalLM.""" ""

    ff_gate: Int8QuantResult


class SmoothQuantLlamaHook(SmoothQuantHook):
    """SmoothQuant Hook for LlamaForCausalLM."""

    def __init__(self, config: PretrainedConfig):
        """Initialize SmoothQuantLlamaHook."""
        super().__init__(config)
        config = cast(LlamaConfig, self.model_config)
        self.num_attention_heads = config.num_attention_heads
        if config.num_key_value_heads is None:
            self.num_kv_attention_heads = self.num_attention_heads
        else:
            self.num_kv_attention_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_dim = self.head_size

    def iter_smooth_norm_weights(
        self, model: LlamaForCausalLM
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in LlamaForCausalLM."""
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
                ],
                f"{self.quantized_layer_prefix}.{index}.mlp.up_proj",  # the input tensors fed into up_proj, gate_proj matrices are identical.
            )
            # [LayerNomr 2] = [ MLP GATED FF ] gets smoothed
            yield (
                [
                    decoder_layer.post_attention_layernorm.weight.data,
                ],
                [
                    decoder_layer.mlp.gate_proj.weight.data,
                ],
                f"{self.quantized_layer_prefix}.{index}.mlp.gate_proj",  # the input tensors fed into up_proj, gate_proj matrices are identical.
            )

    def reshape_qkv_weight(
        self, attn_layer: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape LlamaForCausalLM's qkv weight for int8 quantization."""
        q_weight, k_weight, v_weight = (
            cast(torch.nn.Linear, attn_layer.q_proj).weight,
            cast(torch.nn.Linear, attn_layer.k_proj).weight,
            cast(torch.nn.Linear, attn_layer.v_proj).weight,
        )
        q_weight = q_weight.reshape(
            self.num_attention_heads,
            self.head_size,
            self.hidden_size,
        )
        k_weight = k_weight.reshape(
            self.num_kv_attention_heads,
            self.head_size,
            self.hidden_size,
        )
        q_weight = convert_to_gpt_j_params(q_weight, self.rotary_dim)
        k_weight = convert_to_gpt_j_params(k_weight, self.rotary_dim)
        q_weight = q_weight.reshape(
            self.num_attention_heads * self.head_size,
            self.hidden_size,
        )
        k_weight = k_weight.reshape(
            self.num_kv_attention_heads * self.head_size,
            self.hidden_size,
        )
        return q_weight, k_weight, v_weight

    def sort_qk_output_stats(self, max_output_stat: torch.Tensor) -> torch.Tensor:
        """Sort the output stats of qkv_layer in LlamaForCausalLM."""
        output_stats = max_output_stat.reshape(
            self.num_attention_heads,
            self.head_size,
        )
        output_stats = convert_to_gpt_j_params(output_stats, self.rotary_dim)
        output_stats = output_stats.reshape(
            self.num_attention_heads * self.head_size,
        )
        return output_stats

    def iter_quant_inputs(self, model: LlamaForCausalLM) -> Iterator[TFInt8QuantInputs]:
        """Returns the layers which should be quantized in transformer block of LlamaForCausalLM."""
        for index, decoder_layer in enumerate(model.model.layers):
            self_attn = decoder_layer.self_attn
            q_weight, k_weight, v_weight = self.reshape_qkv_weight(self_attn)
            fc1 = decoder_layer.mlp.up_proj
            ff_gate = decoder_layer.mlp.gate_proj
            fc2 = decoder_layer.mlp.down_proj

            yield LlamaTFInt8QuantInputs(
                layer_index=index,
                q=Int8QuantInput(
                    q_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.q_proj",
                    None,
                    None,
                    self.sort_qk_output_stats,
                ),
                k=Int8QuantInput(
                    k_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.k_proj",
                    None,
                    None,
                    self.sort_qk_output_stats,
                ),
                v=Int8QuantInput(
                    v_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.v_proj",
                    None,
                    None,
                ),
                attn_fc=Int8QuantInput(
                    self_attn.o_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.o_proj",
                    None,
                    None,
                ),
                ff1=Int8QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.up_proj",
                    None,
                    None,
                ),
                ff_gate=Int8QuantInput(
                    ff_gate.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.gate_proj",
                    None,
                    None,
                ),
                ff2=Int8QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.down_proj",
                    None,
                    None,
                ),
            )

    @staticmethod
    def get_quant_result(
        quant_input: TFInt8QuantInputs,
        max_input_stats: Dict[ModuleName, torch.Tensor],
        max_output_stats: Dict[ModuleName, torch.Tensor],
    ) -> TFInt8QuantResults:
        """Returns the quantization result for a specific layer in LlamaForCausalLM."""

        def get_scale(quant_input: Int8QuantInput) -> Int8QuantResult:
            weight, name, start, end = (
                quant_input.weight,
                quant_input.name,
                quant_input.start_offset,
                quant_input.end_offset,
            )
            return get_int8_quant_scales(
                name,
                max_input_stats[name],
                weight[start:end],
                max_output_stats[name][start:end],
            )

        quant_input = cast(LlamaTFInt8QuantInputs, quant_input)
        return LlamaTFInt8QuantResults(
            layer_index=quant_input.layer_index,
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

    @property
    def quantized_layer_prefix(self) -> str:
        """Returns the prefix of the transformer block in LlamaForCausalLM."""
        return "model.layers"
