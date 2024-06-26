# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli LlamaForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Iterator, List, Tuple, Type, cast

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel

from friendli.errors import NotSupportedCheckpointError, QuantizationError
from friendli.modules.quantizer_v2.base import AbstractQuantHookV2
from friendli.modules.quantizer_v2.int8.base import Int8QuantHook
from friendli.modules.quantizer_v2.schema.config import Int8QuantConfig
from friendli.modules.quantizer_v2.schema.data import (
    ModuleName,
    QuantInput,
    TFQuantInputs,
)


class LlamaQuantHook(AbstractQuantHookV2):
    """BaseQuantHook for LlamaForCausalLM."""

    def check_model_config(self) -> None:
        """Check if LLaMA architectures' config can be converted to Friendli format."""
        try:
            if cast(LlamaConfig, self.model_config).hidden_act not in ["silu"]:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'hidden_act={cast(LlamaConfig, self.model_config).hidden_act}'",
                    valid_options=["silu"],
                )
            if cast(LlamaConfig, self.model_config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if cast(LlamaConfig, self.model_config).rms_norm_eps not in (1e-5, 1e-6):
                raise NotSupportedCheckpointError(
                    invalid_option=f"'rms_norm_eps={cast(LlamaConfig, self.model_config).rms_norm_eps}'",
                    valid_options=[1e-5, 1e-6],
                )
        except AttributeError as exc:
            raise QuantizationError(str(exc)) from exc

    def get_tf_blocks(self, model: PreTrainedModel) -> List[torch.nn.Module]:
        """Return the transformer blocks in LlamaForCausalLM."""
        return model.model.layers

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Return the linear layer types in LlamaForCausalLM."""
        return (torch.nn.Linear,)

    @property
    def quantized_layer_prefix(self) -> str:
        """The layer name prefix used before LLaMA's transformer block number."""
        return "model.layers."


class LlamaInt8QuantHook(LlamaQuantHook, Int8QuantHook):
    """Int8QuantHook for LlamaForCausalLM."""

    def get_attn_fc_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Return the linear layer after attention in the decoder layer."""
        return decoder_layer.self_attn.o_proj

    def get_ff2_layer(self, decoder_layer: torch.nn.Module) -> torch.nn.Linear:
        """Return the linear layer after FF1 in the decoder layer."""
        return decoder_layer.mlp.down_proj

    def iter_pre_act_post_act_params(
        self,
        model: LlamaForCausalLM,
    ) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], ModuleName]]:
        """Return iterator of layernorm's weight and linear layer's weight per transformer block in LlamaForCausalLM."""

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
            # [LayerNorm 2] - [ MLP FF 1, MLP FF GATE ] gets smoothed
            yield (
                [
                    decoder_layer.post_attention_layernorm.weight.data,
                ],
                [
                    decoder_layer.mlp.up_proj.weight.data,
                    decoder_layer.mlp.gate_proj.weight.data,
                ],
                f"{self.quantized_layer_prefix}{index}.mlp.up_proj",
            )

    def iter_tf_quant_inputs(self, model: PreTrainedModel) -> Iterator[TFQuantInputs]:
        """Return the layers which should be quantized in transformer block of LlamaForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            self_attn = decoder_layer.self_attn
            mlp = decoder_layer.mlp

            yield TFQuantInputs(
                layer_index=index,
                block=decoder_layer,
                quant_inputs=[
                    QuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.self_attn.q_proj",
                        ],
                        local_names=["q_proj"],
                    ),
                    QuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.self_attn.k_proj",
                        ],
                        local_names=["k_proj"],
                    ),
                    QuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.self_attn.v_proj",
                        ],
                        local_names=["v_proj"],
                    ),
                    QuantInput(
                        parent_module=self_attn,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.self_attn.o_proj",
                        ],
                        local_names=[
                            "o_proj",
                        ],
                    ),
                    QuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.mlp.up_proj",
                        ],
                        local_names=["up_proj"],
                    ),
                    QuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.mlp.gate_proj",
                        ],
                        local_names=["gate_proj"],
                    ),
                    QuantInput(
                        parent_module=mlp,
                        target_names=[
                            f"{self.quantized_layer_prefix}{index}.mlp.down_proj"
                        ],
                        local_names=["down_proj"],
                    ),
                ],
            )
