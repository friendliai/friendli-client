# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli LlamaForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Tuple, Type, cast

import torch

from friendli.modules.converter.base import DECODER_PREFIX
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.quantizer.awq.base import AWQHook
from friendli.modules.quantizer.schema.config import AWQConfig
from friendli.modules.quantizer.schema.data import (
    ModuleName,
    QuantInput,
    TFQuantInputs,
    TFQuantResults,
    WeightOnlyQuantResult,
)
from friendli.modules.quantizer.utils import (
    get_weight_only_quant_scales,
    quantized_linear_weight_reshape,
    scale_reshape,
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
        self.scale_attn_fc = self.num_attention_heads == self.num_kv_attention_heads

    def add_pre_scaler(self, model: torch.nn.Module) -> torch.nn.Module:
        """Adds scaler to LlamaForCausalLM."""
        return model

    def get_inspect_module_types(
        self, block: torch.nn.Module
    ) -> Tuple[type[torch.nn.Module], ...]:
        """Returns the layer types in inspected blocks."""
        return (type(block.self_attn), type(block.mlp))

    def iter_inspect_modules(
        self,
        block: torch.nn.Module,
    ) -> Iterator[
        Tuple[
            List[torch.nn.Module],
            List[Tuple[ModuleName, torch.nn.Linear]],
            torch.nn.Module,
            ModuleName,
        ]
    ]:
        """Returns iterator of layers in blocks."""
        # qkv proj
        yield (
            [block.input_layernorm],
            [
                ("self_attn.q_proj", block.self_attn.q_proj),
                ("self_attn.k_proj", block.self_attn.k_proj),
                ("self_attn.v_proj", block.self_attn.v_proj),
            ],
            block.self_attn,
            "self_attn",
        )
        # attn out proj
        if self.scale_attn_fc:
            yield (
                [block.self_attn.v_proj],
                [("self_attn.o_proj", block.self_attn.o_proj)],
                block.self_attn.o_proj,
                "self_attn.o_proj",
            )
        # ff1
        yield (
            [block.post_attention_layernorm],
            [
                ("mlp.up_proj", block.mlp.up_proj),
                ("mlp.gate_proj", block.mlp.gate_proj),
            ],
            block.mlp,
            "mlp",
        )
        # ff2
        yield (
            [block.mlp.up_proj],
            [("mlp.down_proj", block.mlp.down_proj)],
            block.mlp.down_proj,
            "mlp.down_proj",
        )

    def iter_tf_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of LlamaForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.self_attn
            q_weight, k_weight, v_weight = (
                self.converter.qkv_weight_reshape(
                    [
                        self_attn.q_proj.weight,
                        self_attn.k_proj.weight,
                        self_attn.v_proj.weight,
                    ]
                )
                .transpose(0, 1)
                .split(
                    [
                        self.converter.decoder_num_attention_heads
                        * self.converter.decoder_head_size,
                        self.converter.decoder_num_kv_attention_heads
                        * self.converter.decoder_head_size,
                        self.converter.decoder_num_kv_attention_heads
                        * self.converter.decoder_head_size,
                    ],
                    dim=0,
                )
            )
            fc1 = decoder_layer.mlp.up_proj
            ff_gate = decoder_layer.mlp.gate_proj
            fc2 = decoder_layer.mlp.down_proj

            yield LlamaTFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                q=QuantInput(
                    q_weight,
                    f"{self.quantized_layer_prefix}{index}.self_attn.q_proj",
                    None,
                    None,
                ),
                k=QuantInput(
                    k_weight,
                    f"{self.quantized_layer_prefix}{index}.self_attn.k_proj",
                    None,
                    None,
                ),
                v=QuantInput(
                    v_weight,
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
        """Get quantization result for a specific layer in LlamaForCausalLM."""
        awq_config = cast(AWQConfig, self.quant_config)

        def get_scale(quant_input: QuantInput) -> WeightOnlyQuantResult:
            weight, name, start, end = (
                quant_input.weight,
                quant_input.name,
                quant_input.start_offset,
                quant_input.end_offset,
            )
            weight = weight.to(awq_config.device)

            return get_weight_only_quant_scales(
                layer_name=name,
                w=weight[start:end],
                q_bit=awq_config.awq_args.quant_bit,
                q_group_size=awq_config.awq_args.quant_group_size,
            )

        quant_input = cast(LlamaTFQuantInputs, quant_input)
        return LlamaTFQuantResults(
            layer_prefix_with_index=f"{self.quantized_layer_prefix}{quant_input.layer_index}.",
            block=quant_input.block,
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
        param_names = super().quantized_param_names
        for i in range(self.converter.decoder_layer_num):
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            param_names.append(
                f"{converted_prefix}mlp/c_gate/weight:0",
            )
        return param_names

    @property
    def modified_layers_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for modified layers."""
        return []

    @property
    def avoid_clipping_layer_names(self) -> List[str]:
        """Returns the layer names which should be avoided for AWQ clipping."""
        return ["q_proj", "k_proj"]

    @property
    def quantized_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the convert_info_list for quantized layers."""
        convert_info_list = super().quantized_convert_info_list
        for i in range(self.converter.decoder_layer_num):
            layer_prefix = f"{self.quantized_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"

            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff_gate.weight_scale"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}mlp/c_gate/awq/scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff_gate.zeros"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}mlp/c_gate/awq/zero:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ff_gate.weight"],
                        data_type=self.quant_dtype,
                        converted_name=f"{converted_prefix}mlp/c_gate/awq/weight:0",
                        reshape_fn=quantized_linear_weight_reshape,
                    ),
                ]
            )
        return convert_info_list
