# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPTJForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Tuple, Type

import numpy as np
import torch

from periflow.modules.converter.utils import nontype_partial
from periflow.modules.quantizer.awq.base import AWQHook
from periflow.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from periflow.modules.quantizer.utils import scale_convert


class AWQGPTJHook(AWQHook):
    """AWQ Hook for GPTJForCausalLM."""

    def __init__(self, quant_config, converter):
        """Initialize AWQGPTJHook."""
        super().__init__(quant_config, converter)
        config = converter.config
        self.data_type = converter.data_type
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_dim = config.rotary_dim

    def add_pre_scaler(self, model: torch.nn.Module) -> torch.nn.Module:
        """Adds scaler to GPTJForCausalLM."""
        for tf_block in self.get_tf_blocks(model):
            ff2_scaler = self._register_pre_scaler(tf_block.mlp.fc_out)
            tf_block.mlp.add_module("ff2_scaler", ff2_scaler)
        return model

    def get_inspect_module_types(
        self, block: torch.nn.Module
    ) -> Tuple[Type[torch.nn.Module], ...]:
        """Returns the type of linear layer (etc. qkv, linear layer) in transformer block."""
        return (type(block.attn), type(block.mlp))

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
            [block.ln_1],
            [
                ("attn.q_proj", block.attn.q_proj),
                ("attn.k_proj", block.attn.k_proj),
                ("attn.v_proj", block.attn.v_proj),
                ("mlp.fc_in", block.mlp.fc_in),
            ],
            block,
            "",
        )
        # attn out proj
        yield (
            [block.attn.v_proj],
            [("attn.out_proj", block.attn.out_proj)],
            block.attn.out_proj,
            "attn.out_proj",
        )
        # ff2
        yield (
            [block.mlp.ff2_scaler],
            [("mlp.fc_out", block.mlp.fc_out)],
            block.mlp.fc_out,
            "mlp.fc_out",
        )

    def iter_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of GPTJForCausalLM."""
        for index, tf_block in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            yield TFQuantInputs(
                layer_index=index,
                q=QuantInput(
                    tf_block.attn.q_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.q_proj",
                    None,
                    None,
                ),
                k=QuantInput(
                    tf_block.attn.k_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.k_proj",
                    None,
                    None,
                ),
                v=QuantInput(
                    tf_block.attn.v_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.v_proj",
                    None,
                    None,
                ),
                attn_fc=QuantInput(
                    tf_block.attn.out_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.out_proj",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    tf_block.mlp.fc_in.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.fc_in",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    tf_block.mlp.fc_out.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.fc_out",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in GPTJForCausalLM."""
        return (torch.nn.Linear,)

    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the transformer blocks in GPTJForCausalLM."""
        return model.transformer.h  # type: ignore

    @property
    def modified_layers_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for modified layers."""
        return {
            "mlp/c_proj/awq/pre_scale:0": nontype_partial(
                scale_convert,
                per_layer_postfixes=[".mlp.ff2_scaler.scale"],
                data_type="fp32",
            ),
        }

    @property
    def avoid_clipping_layer_names(self) -> List[str]:
        """Returns the layer names which should be avoided for AWQ clipping."""
        return ["q_proj", "k_proj"]
