# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow MPTForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Tuple, Type

import numpy as np
import torch

from periflow.modules.converter.utils import nontype_partial
from periflow.modules.quantizer.awq.base import AWQHook
from periflow.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from periflow.modules.quantizer.utils import scale_convert


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

    def iter_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of MPTForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.attn
            attn_weight_outdim = self_attn.Wqkv.weight.size(0)  # OutDim
            fc1 = decoder_layer.ffn.up_proj  # type: ignore
            fc2 = decoder_layer.ffn.down_proj  # type: ignore

            yield TFQuantInputs(
                layer_index=index,
                parent_module=decoder_layer,
                q=QuantInput(
                    self_attn.Wqkv.weight,  # type: ignore
                    f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                    0,
                    attn_weight_outdim // 3,
                ),
                k=QuantInput(
                    self_attn.Wqkv.weight,  # type: ignore
                    f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                    attn_weight_outdim // 3,
                    attn_weight_outdim // 3 * 2,
                ),
                v=QuantInput(
                    self_attn.Wqkv.weight,  # type: ignore
                    f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                    attn_weight_outdim // 3 * 2,
                    attn_weight_outdim,
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
    def modified_layers_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for modified layers."""
        return {
            "attn/c_proj/awq/pre_scale:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".attn.scaler.scale"],
                data_type="fp32",
            ),
            "mlp/c_proj/awq/pre_scale:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".ffn.scaler.scale"],
                data_type="fp32",
            ),
        }

    @property
    def avoid_clipping_layer_names(self) -> List[str]:
        """Returns the layer names which should be avoided for AWQ clipping."""
        return ["Wqkv"]
