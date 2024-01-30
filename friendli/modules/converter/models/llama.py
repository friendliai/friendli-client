# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli LLaMAa Checkpoint Converter."""


from __future__ import annotations

from typing import Any, Dict, List, cast

import torch
from transformers import LlamaConfig, LlamaForCausalLM  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import (
    DECODER_PREFIX,
    DecoderOnlyConverter,
    DecoderOnlyLoraConverter,
)
from friendli.modules.converter.interface import RotaryEmbeddingConversionInterface
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.converter.utils import convert_to_gpt_j_params


class LlamaForCausalLMLoraConverter(DecoderOnlyLoraConverter):
    """LlamaForCausalLM LoRA Converter Class."""

    def pre_convert(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        """Adjust the LoRA Adapter module's params in Llama before converting."""
        converter = cast(LlamaForCausalLMConverter, self.converter)
        for layer in cast(LlamaForCausalLM, model).model.layers:
            query_b = layer.self_attn.q_proj.lora_B.default.weight
            query_b = query_b.reshape(
                converter.decoder_num_attention_heads,
                converter.decoder_head_size,
                -1,
            )
            query_b = convert_to_gpt_j_params(query_b, converter.decoder_head_size)
            query_b = query_b.reshape(
                converter.decoder_num_attention_heads * converter.decoder_head_size,
                -1,
            )
            layer.self_attn.q_proj.lora_B.default.weight.data = query_b
        return model

    @property
    def adapter_target_modules(self) -> List[str]:
        """Return the target modules that LoRA applies to."""
        return ["query", "value"]

    @property
    def adapter_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for LoRA adapter modules in Llama."""
        convert_info_list = []
        for i in range(self.converter.decoder_layer_num):
            layer_prefix = f"{self.converter.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/lora/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}self_attn.q_proj.lora_A.default.weight"
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}query_A/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}self_attn.q_proj.lora_B.default.weight"
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}query_B/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}self_attn.v_proj.lora_A.default.weight"
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}value_A/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}self_attn.v_proj.lora_B.default.weight"
                        ],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}value_B/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                ]
            )
        return convert_info_list


class LlamaForCausalLMConverter(
    DecoderOnlyConverter, RotaryEmbeddingConversionInterface
):
    """LlamaForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if LLaMA architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if cast(LlamaConfig, self.config).hidden_act not in ["silu"]:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'hidden_act={cast(LlamaConfig, self.config).hidden_act}'",
                    valid_options=["silu"],
                )
            if cast(LlamaConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if cast(LlamaConfig, self.config).rms_norm_eps not in (1e-5, 1e-6):
                raise NotSupportedCheckpointError(
                    invalid_option=f"'rms_norm_eps={cast(LlamaConfig, self.config).rms_norm_eps}'",
                    valid_options=[1e-5, 1e-6],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def qkv_weight_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """qkv_weight_reshape for LLaMA's attention layer."""
        assert len(params) == 3
        q_weight = params[0]
        k_weight = params[1]
        v_weight = params[2]

        q_weight = q_weight.reshape(
            self.decoder_num_attention_heads,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )
        k_weight = k_weight.reshape(
            self.decoder_num_kv_attention_heads,
            self.decoder_head_size,
            self.decoder_hidden_size,
        )
        q_weight = convert_to_gpt_j_params(q_weight, self.rotary_dim)
        k_weight = convert_to_gpt_j_params(k_weight, self.rotary_dim)
        q_weight = q_weight.reshape(
            self.decoder_num_attention_heads * self.decoder_head_size,
            self.decoder_hidden_size,
        )
        k_weight = k_weight.reshape(
            self.decoder_num_kv_attention_heads * self.decoder_head_size,
            self.decoder_hidden_size,
        )

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        qkv_weight = qkv_weight.transpose(0, -1)
        return qkv_weight

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(LlamaConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The Llama model does not rely on "
            "absolute position embeddings, allowing you to choose any suitable value.",
            config.max_position_embeddings,
        )

        eos_token_id = self.get_eos_token_id()
        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.decoder_head_size,
            "rotary_dim": self.rotary_dim,
            "num_heads": self.decoder_num_attention_heads,
            "num_kv_heads": self.decoder_num_kv_attention_heads,
            "num_layers": self.decoder_layer_num,
            "ff_intermediate_size": self.decoder_ff_intermediate_size,
            "max_length": config.max_position_embeddings,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
            "rope_theta": self.rotary_emb_base,
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "llama"

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in LLaMA."""
        convert_info_list = []
        for i in range(self.decoder_layer_num):
            layer_prefix = f"{self.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}input_layernorm.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[
                            f"{layer_prefix}self_attn.q_proj.weight",
                            f"{layer_prefix}self_attn.k_proj.weight",
                            f"{layer_prefix}self_attn.v_proj.weight",
                        ],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/weight:0",
                        reshape_fn=self.qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}self_attn.o_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}post_attention_layernorm.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.gate_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_gate/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.up_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.down_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                ]
            )
        return convert_info_list

    @property
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for non-transformer blocks in LLaMA."""
        return [
            ConvertInfo(
                param_names=["model.embed_tokens.weight"],
                data_type=self.data_type,
                converted_name="wte/weight:0",
                reshape_fn=self.token_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["model.norm.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/gamma:0",
                reshape_fn=self.ln_weight_reshape,
            ),
            ConvertInfo(
                param_names=["lm_head.weight"],
                data_type=self.data_type,
                converted_name=f"head_fc/weight:0",
                reshape_fn=self.head_weight_reshape,
            ),
        ]

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before LLaMA's transformer block number."""
        return "model.layers."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in LLaMA."""
        return cast(LlamaConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in LLaMA."""
        return cast(LlamaConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in LLaMA."""
        return cast(LlamaConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in LLaMA."""
        config = cast(LlamaConfig, self.config)
        if config.num_key_value_heads is None:
            return self.decoder_num_attention_heads
        return config.num_key_value_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of LLaMA."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in LLaMA MLP."""
        return self.config.intermediate_size

    @property
    def rotary_dim(self) -> int:
        """The rotary embedding dimension of LLaMA."""
        return self.decoder_head_size

    @property
    def rotary_emb_base(self) -> float:
        """The rotary embedding base of LLaMA."""
        return cast(LlamaConfig, self.config).rope_theta
