# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli MPTForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Iterator, List, Tuple, Type

import torch

from friendli.enums import CheckpointDataType
from friendli.modules.converter.base import DECODER_PREFIX
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.quantizer.awq.base import AWQHook
from friendli.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from friendli.modules.quantizer.utils import scale_reshape


class AWQMPTHook(AWQHook):
    """AWQ Hook for MPTForCausalLM."""

    def add_pre_scaler(self, model: torch.nn.Module) -> torch.nn.Module:
        """Adds scaler to MPTForCausalLM."""
        for tf_block in self.get_tf_blocks(model):
            attn_fc_scaler = self._register_pre_scaler(
                tf_block.attn.out_proj,
            )
            tf_block.attn.add_module("scaler", attn_fc_scaler)
            ff2_scaler = self._register_pre_scaler(tf_block.ffn.down_proj)
            tf_block.ffn.add_module("scaler", ff2_scaler)
        return model

    def get_inspect_module_types(
        self, block: torch.nn.Module
    ) -> Tuple[Type[torch.nn.Module], ...]:
        """Returns the type of linear layer (etc. qkv, linear layer) in transformer block."""
        return (type(block.attn), type(block.ffn))

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
        """Returns iterator of layers in modules."""
        # qkv proj
        yield (
            [block.norm_1],
            [("attn.Wqkv", block.attn.Wqkv)],
            block.attn,
            "attn",
        )
        # attn out proj
        yield (
            [block.attn.scaler],
            [("attn.out_proj", block.attn.out_proj)],
            block.attn.out_proj,
            "attn.out_proj",
        )
        # ff1
        yield (
            [block.norm_2],
            [("ffn.up_proj", block.ffn.up_proj)],
            block.ffn,
            "ffn",
        )
        # ff2
        yield (
            [block.ffn.scaler],
            [("ffn.down_proj", block.ffn.down_proj)],
            block.ffn.down_proj,
            "ffn.down_proj",
        )

    def iter_tf_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of MPTForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.attn
            q_outdim = (
                self.converter.decoder_num_attention_heads
                * self.converter.decoder_head_size
            )
            kv_outdim = (
                self.converter.decoder_num_kv_attention_heads
                * self.converter.decoder_head_size
            )
            qkv_outdim = self_attn.Wqkv.weight.size(0)
            assert qkv_outdim == q_outdim + kv_outdim * 2
            fc1 = decoder_layer.ffn.up_proj  # type: ignore
            fc2 = decoder_layer.ffn.down_proj  # type: ignore

            yield TFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                q=QuantInput(
                    self_attn.Wqkv.weight,  # type: ignore
                    f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                    0,
                    q_outdim,
                ),
                k=QuantInput(
                    self_attn.Wqkv.weight,  # type: ignore
                    f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                    q_outdim,
                    q_outdim + kv_outdim,
                ),
                v=QuantInput(
                    self_attn.Wqkv.weight,  # type: ignore
                    f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                    q_outdim + kv_outdim,
                    qkv_outdim,
                ),
                attn_fc=QuantInput(
                    self_attn.out_proj.weight,  # type: ignore
                    f"{self.quantized_layer_prefix}{index}.attn.out_proj",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    fc1.weight,  # type: ignore
                    f"{self.quantized_layer_prefix}{index}.ffn.up_proj",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    fc2.weight,  # type: ignore
                    f"{self.quantized_layer_prefix}{index}.ffn.down_proj",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in MPTForCausalLM."""
        return (torch.nn.Linear,)

    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the transformer blocks in MPTForCausalLM."""
        return model.transformer.blocks  # type: ignore

    @property
    def modified_layers_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for modified layers."""
        convert_info_list = []
        for i in range(self.converter.decoder_layer_num):
            layer_prefix = f"{self.quantized_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn.scaler.scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_proj/awq/pre_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ffn.scaler.scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}mlp/c_proj/awq/pre_scale:0",
                        reshape_fn=scale_reshape,
                    ),
                ]
            )
        return convert_info_list

    @property
    def avoid_clipping_layer_names(self) -> List[str]:
        """Returns the layer names which should be avoided for AWQ clipping."""
        return ["Wqkv"]
