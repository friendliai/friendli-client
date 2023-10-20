# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CodeGenForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

import copy
from typing import Callable, Dict, Iterator, List, Tuple, Type, cast

import numpy as np
import torch
from transformers.models.codegen import CodeGenForCausalLM  # type: ignore[import]

from periflow.modules.converter.utils import nontype_partial
from periflow.modules.quantizer.schema.config import SmoothQuantConfig
from periflow.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from periflow.modules.quantizer.smoothquant.base import SmoothQuantHook


class SmoothQuantCodeGenHook(SmoothQuantHook):
    """SmoothQuant Hook for CodeGenForCausalLM."""

    def pre_smooth(self, model: torch.nn.Module) -> torch.nn.Module:
        """Pre-procedure for SmoothQuant in CodeGenForCausalLM that should be called before smooth() is called."""
        super().pre_smooth(model)
        for decoder_layer in cast(CodeGenForCausalLM, model).transformer.h:
            decoder_layer.add_module("ln_2", copy.deepcopy(decoder_layer.ln_1))
        return model

    def iter_smooth_norm_weights(
        self,
        model: CodeGenForCausalLM,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Returns iterator of layernorm's weight and linear layer's weight per transformer block in CodeGenForCausalLM."""
        quant_args = cast(SmoothQuantConfig, self.quant_config).smoothquant_args

        for index, decoder_layer in enumerate(model.transformer.h):  # type: ignore[union-attr]
            # [LayerNorm 1] - [ QKV projection, MLP FF 1 ] gets smoothed
            yield (
                [
                    decoder_layer.ln_1.weight.data,
                    decoder_layer.ln_1.bias.data,
                ],
                [
                    decoder_layer.attn.qkv_proj.weight.data,
                ],
                f"{self.quantized_layer_prefix}{index}.attn.qkv_proj",
            )
            yield (
                [
                    decoder_layer.ln_2.weight.data,
                    decoder_layer.ln_2.bias.data,
                ],
                [
                    decoder_layer.mlp.fc_in.weight.data,
                ],
                f"{self.quantized_layer_prefix}{index}.mlp.fc_in",
            )
            if quant_args.attn_fc_smoothing:
                yield (
                    [decoder_layer.attn_fc_pre_smoother.scale.data],
                    [decoder_layer.attn.out_proj.weight.data],
                    f"{self.quantized_layer_prefix}{index}.self_attention.dense",
                )
            if quant_args.ff2_smoothing:
                yield (
                    [decoder_layer.ff2_pre_smoother.scale.data],
                    [decoder_layer.mlp.fc_out.data],
                    f"{self.quantized_layer_prefix}{index}.mlp.dense_4h_to_h",
                )

    def reshape_qkv_weight(
        self, attn_layer: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshapes the qkv weight in CodeGenForCausalLM for Quantization."""
        qkv_layer = cast(torch.nn.Linear, attn_layer.qkv_proj)
        original_qkv_weight = qkv_layer.weight
        reshaped_qkv_weight = original_qkv_weight.reshape(
            (4, original_qkv_weight.size(0) // 4, original_qkv_weight.size(1))
        )
        q_weight, v_weight, k_weight = torch.split(
            reshaped_qkv_weight, reshaped_qkv_weight.size(1) // 3, dim=1
        )
        q_weight = q_weight.reshape((-1, q_weight.size(2)))
        k_weight = k_weight.reshape((-1, k_weight.size(2)))
        v_weight = v_weight.reshape((-1, v_weight.size(2)))

        return q_weight, k_weight, v_weight

    def sort_qkv_output_stats(self, max_output_stat: torch.Tensor) -> torch.Tensor:
        """Sorts the max output stats of qkv_proj in CodeGenForCausalLM."""
        reshpaed_max_output_stat = max_output_stat.reshape(
            (4, max_output_stat.size(0) // 4)
        )
        q_max_output_stat, v_max_output_stat, k_max_output_stat = torch.split(
            reshpaed_max_output_stat, reshpaed_max_output_stat.size(1) // 3, dim=1
        )
        q_max_output_stat = q_max_output_stat.reshape((-1,))
        k_max_output_stat = k_max_output_stat.reshape((-1,))
        v_max_output_stat = v_max_output_stat.reshape((-1,))
        return torch.cat(
            (q_max_output_stat, k_max_output_stat, v_max_output_stat), dim=0
        )

    def iter_quant_inputs(self, model: CodeGenForCausalLM) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of CodeGenForCausalLM."""
        for index, decoder_layer in enumerate(model.transformer.h):
            self_attn = decoder_layer.attn
            q_weight, k_weight, v_weight = self.reshape_qkv_weight(self_attn)
            qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            attn_weight_outdim = qkv_weight.size(0)  # OutDim
            fc1 = decoder_layer.mlp.fc_in
            fc2 = decoder_layer.mlp.fc_out

            yield TFQuantInputs(
                layer_index=index,
                q=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.attn.qkv_proj",
                    0,
                    attn_weight_outdim // 3,
                    self.sort_qkv_output_stats,
                ),
                k=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.attn.qkv_proj",
                    attn_weight_outdim // 3,
                    attn_weight_outdim // 3 * 2,
                    self.sort_qkv_output_stats,
                ),
                v=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.attn.qkv_proj",
                    attn_weight_outdim // 3 * 2,
                    attn_weight_outdim,
                    self.sort_qkv_output_stats,
                ),
                attn_fc=QuantInput(
                    self_attn.out_proj.weight,
                    f"{self.quantized_layer_prefix}{index}.attn.out_proj",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    fc1.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.fc_in",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    fc2.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.fc_out",
                    None,
                    None,
                ),
            )

    @property
    def modified_layers_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Returns the convert_dict for modified layers in CodeGenForCausalLM."""
        convert_dict = super().modified_layers_convert_dict
        convert_dict.update(
            {
                "ln_2/gamma:0": nontype_partial(
                    self.converter.ln_weight_convert,
                    per_layer_postfixes=[".ln_2.weight"],
                ),
                "ln_2/beta:0": nontype_partial(
                    self.converter.ln_bias_convert,
                    per_layer_postfixes=[".ln_2.bias"],
                ),
            }
        )
        return convert_dict

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in CodeGenForCausalLM."""
        return (torch.nn.Linear,)

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after attention in the decoder layer."""
        return decoder_layer.attn.out_proj

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Returns the linear layer after FF1 in the decoder layer."""
        return decoder_layer.mlp.fc_out

    def get_tf_blocks(self, model: CodeGenForCausalLM) -> List[torch.nn.Module]:
        """Returns the decoder layers(transformer blocks) in the model."""
        return list(model.transformer.h)
