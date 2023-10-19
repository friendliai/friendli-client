# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow LlamaForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, cast

import numpy as np
import torch

from periflow.modules.converter.utils import convert_to_gpt_j_params, nontype_partial
from periflow.modules.quantizer.awq.base import AWQHook
from periflow.modules.quantizer.schema.config import AWQConfig
from periflow.modules.quantizer.schema.data import (
    ModuleName,
    QuantInput,
    TFQuantInputs,
    TFQuantResults,
    WeightOnlyQuantResult,
)
from periflow.modules.quantizer.utils import (
    get_weight_only_quant_scales,
    quantized_linear_weight_convert,
    scale_convert,
)


@dataclass
class LlamaTFQuantInputs(TFQuantInputs):
    """Dataclass for quantization input per layer in LlamaForCausalLM."""

    ff_gate: QuantInput


@dataclass
class LlamaTFQuantResults(TFQuantResults):
    """Dataclass for quantization result per layer in LlamaForCausalLM."""

    ff_gate: WeightOnlyQuantResult


class AWQLlamaHook(AWQHook):
    """AWQ Hook for LlamaForCausalLM."""

    def __init__(self, quant_config, converter):
        """Initialize AWQLlamaHook."""
        super().__init__(quant_config, converter)
        config = converter.config
        self.data_type = converter.data_type
        self.num_attention_heads = config.num_attention_heads
        if config.num_key_value_heads is None:
            self.num_kv_attention_heads = self.num_attention_heads
        else:
            self.num_kv_attention_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_dim = self.head_size
        self.require_attn_fc_pre_scale = (
            self.num_attention_heads != self.num_kv_attention_heads
        )

    def add_pre_scaler(self, model: torch.nn.Module) -> torch.nn.Module:
        """Adds scaler to LlamaForCausalLM."""
        for tf_block in self.get_tf_blocks(model):
            if self.require_attn_fc_pre_scale:
                attn_fc_scaler = self._register_pre_scaler(
                    tf_block.self_attn.o_proj,
                )
                tf_block.self_attn.add_module("scaler", attn_fc_scaler)
        return model

    def get_inspect_module_types(
        self, block: torch.nn.Module
    ) -> Tuple[type[torch.nn.Module], ...]:
        """Returns the layer types in inspected blocks."""
        return (type(block.self_attn),)

    def iter_inspect_modules(
        self,
        block: torch.nn.Module,
        intra_kwargs: Dict[str, Any],
    ) -> Iterator[
        Tuple[
            List[torch.nn.Module],
            List[torch.nn.Module],
            ModuleName,
            torch.nn.Module,
            Dict[str, Any],
        ]
    ]:
        """Returns iterator of layers in blocks."""
        # qkv proj
        yield (
            [block.input_layernorm],
            [
                block.self_attn.q_proj,
                block.self_attn.k_proj,
                block.self_attn.v_proj,
            ],
            "self_attn.q_proj",  # "self_attn.k_proj" or "self_attn.v_proj" is also valid
            block.self_attn,
            intra_kwargs["self_attn"],
        )
        # attn out proj
        yield (
            [
                block.self_attn.scaler
                if self.require_attn_fc_pre_scale
                else block.self_attn.v_proj
            ],
            [block.self_attn.o_proj],
            "self_attn.o_proj",
            block.self_attn.o_proj,
            {},
        )
        # ff1
        yield (
            [block.post_attention_layernorm],
            [
                block.mlp.up_proj,
                block.mlp.gate_proj,
            ],
            "mlp.gate_proj",  # "mlp.up_proj" is also valid
            block.mlp,
            {},
        )
        # ff2
        yield (
            [block.mlp.up_proj],
            [block.mlp.down_proj],
            "mlp.down_proj",
            block.mlp.down_proj,
            {},
        )

    def reshape_qkv_weight(
        self, attn_layer: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape LlamaForCausalLM's qkv weight."""
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

    def iter_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of LlamaForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.self_attn
            q_weight, k_weight, v_weight = self.reshape_qkv_weight(self_attn)
            fc1 = decoder_layer.mlp.up_proj
            ff_gate = decoder_layer.mlp.gate_proj
            fc2 = decoder_layer.mlp.down_proj

            yield LlamaTFQuantInputs(
                layer_index=index,
                q=QuantInput(
                    q_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.q_proj",
                    None,
                    None,
                ),
                k=QuantInput(
                    k_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.k_proj",
                    None,
                    None,
                ),
                v=QuantInput(
                    v_weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.v_proj",
                    None,
                    None,
                ),
                attn_fc=QuantInput(
                    self_attn.o_proj.weight,
                    f"{self.quantized_layer_prefix}.{index}.self_attn.o_proj",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.up_proj",
                    None,
                    None,
                ),
                ff_gate=QuantInput(
                    ff_gate.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.gate_proj",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}.{index}.mlp.down_proj",
                    None,
                    None,
                ),
            )

    def get_quant_result(
        self,
        quant_input: TFQuantInputs,
        **kwargs: Any,
    ) -> TFQuantResults:
        """Get quantization result for a specific layer in LlamaForCausalLM."""
        awq_args = cast(AWQConfig, self.quant_config).awq_args

        def get_scale(quant_input: QuantInput) -> WeightOnlyQuantResult:
            weight, name, start, end = (
                quant_input.weight,
                quant_input.name,
                quant_input.start_offset,
                quant_input.end_offset,
            )

            return get_weight_only_quant_scales(
                layer_name=name,
                w=weight[start:end],
                q_bit=awq_args.quant_bit,
                q_group_size=awq_args.quant_group_size,
            )

        quant_input = cast(LlamaTFQuantInputs, quant_input)
        return LlamaTFQuantResults(
            layer_prefix_with_index=f"{self.quantized_layer_prefix}.{quant_input.layer_index}",
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

    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the transformer blocks in LlamaForCausalLM."""
        return model.model.layers

    @property
    def quantized_param_names(self) -> List[str]:
        """Returns the parameter names in LlamaForCausalLM."""
        return super().quantized_param_names + [
            "mlp/c_gate/weight:0",
        ]

    @property
    def modified_layers_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for modified layers."""
        return (
            {
                "attn/c_proj/awq/pre_scale:0": nontype_partial(
                    scale_convert,
                    per_layer_postfixes=[".self_attn.scaler.scale"],
                    data_type="fp32",
                ),
            }
            if self.require_attn_fc_pre_scale
            else {}
        )

    @property
    def avoid_clipping_layer_names(self) -> List[str]:
        """Returns the layer names which should avoid AWQ clipping."""
        return ["q_proj", "k_proj", "v_proj"]

    @property
    def quantized_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for quantized layers."""
        n_bit = cast(AWQConfig, self.quant_config).awq_args.quant_bit
        convert_dict = super().quantized_convert_dict
        convert_dict.update(
            {
                "mlp/c_gate/awq/scale:0": nontype_partial(
                    scale_convert,
                    per_layer_postfixes=[".ff_gate.woq_weight_scale"],
                    data_type=self.data_type,
                ),
                "mlp/c_gate/awq/zero:0": nontype_partial(
                    scale_convert,
                    per_layer_postfixes=[".ff_gate.woq_weight_zp"],
                    data_type=self.data_type,
                ),
                "mlp/c_gate/awq/weight:0": nontype_partial(
                    quantized_linear_weight_convert,
                    per_layer_postfixes=[".ff_gate.woq_weight"],
                    n_bit=n_bit,
                ),
            }
        )
        return convert_dict
