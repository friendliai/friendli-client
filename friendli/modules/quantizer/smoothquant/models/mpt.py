# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli MPTForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Iterator, List, Tuple, Type, cast

import torch

from friendli.modules.quantizer.schema.config import SmoothQuantConfig
from friendli.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from friendli.modules.quantizer.smoothquant.base import SmoothQuantHook


class SmoothQuantMPTHook(SmoothQuantHook):
    """SmoothQuant Hook for MPTForCausalLM."""

    def iter_smooth_norm_weights(
        self,
        model: torch.nn.Module,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in MPTForCausalLM."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args

        for index, decoder_layer in enumerate(
            model.transformer.blocks  # type: ignore[union-attr, arg-type]
        ):
            # [LayerNorm 1] - [ QKV projection ] gets smoothed
            yield (
                [decoder_layer.norm_1.weight.data],
                [decoder_layer.attn.Wqkv.weight.data],
                f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
            )
            # [LayerNorm 2] - [ MLP FF 1 ] gets smoothed
            yield (
                [decoder_layer.norm_2.weight.data],
                [decoder_layer.ffn.up_proj.weight.data],  # [OutDim, InDim]
                f"{self.quantized_layer_prefix}{index}.ffn.up_proj",
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
                    [decoder_layer.ffn.down_proj.weight.data],
                    f"{self.quantized_layer_prefix}{index}.ffn.down_proj",
                )

    def iter_tf_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of MPTForCausalLM."""
        for index, decoder_layer in enumerate(
            model.transformer.blocks  # type: ignore[union-attr, arg-type]
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
            fc1 = decoder_layer.ffn.up_proj
            fc2 = decoder_layer.ffn.down_proj

            yield TFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                q=QuantInput(
                    self_attn.Wqkv.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                    0,
                    q_outdim,
                ),
                k=QuantInput(
                    self_attn.Wqkv.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                    q_outdim,
                    q_outdim + kv_outdim,
                ),
                v=QuantInput(
                    self_attn.Wqkv.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.Wqkv",
                    q_outdim + kv_outdim,
                    qkv_outdim,
                ),
                attn_fc=QuantInput(
                    self_attn.out_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.out_proj",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}{index}.ffn.up_proj",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}{index}.ffn.down_proj",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in MPTForCausalLM."""
        return (torch.nn.Linear,)

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after attention in the decoder layer."""
        return decoder_layer.attn.out_proj

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after FF1 in the decoder layer."""
        return decoder_layer.ffn.down_proj

    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the decoder layers(transformer blocks) in the model."""
        return list(model.transformer.blocks)
