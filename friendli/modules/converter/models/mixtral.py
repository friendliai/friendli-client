# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli Mixtral Checkpoint Converter."""


from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from transformers import MixtralConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import DECODER_PREFIX
from friendli.modules.converter.models.llama import LlamaForCausalLMConverter
from friendli.modules.converter.schema import ConvertInfo


class MixtralForCausalLMConverter(LlamaForCausalLMConverter):
    """MixtralForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Mixtral architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if cast(MixtralConfig, self.config).hidden_act not in ["silu"]:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'hidden_act={cast(MixtralConfig, self.config).hidden_act}'",
                    valid_options=["silu"],
                )
            if cast(MixtralConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if cast(MixtralConfig, self.config).num_local_experts != 8:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'num_local_experts={cast(MixtralConfig, self.config).num_local_experts}",
                    valid_options=[8],
                )
            if cast(MixtralConfig, self.config).num_experts_per_tok != 2:
                raise NotSupportedCheckpointError(
                    invalid_option=f"'num_experts_per_tok={cast(MixtralConfig, self.config).num_experts_per_tok}",
                    valid_options=[2],
                )

        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(MixtralConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The Mixtral model does not rely on "
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
            "num_experts": self.num_experts,
        }
        if isinstance(self.attention_window_size, int):
            # for sliding window
            attr["attention_window_size"] = self.attention_window_size
        return attr

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
                        param_names=[f"{layer_prefix}block_sparse_moe.gate.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}moe/router/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                ]
            )
            for i in range(self.num_experts):
                convert_info_list.extend(
                    [
                        ConvertInfo(
                            param_names=[
                                f"{layer_prefix}block_sparse_moe.experts.{i}.w1.weight"
                            ],
                            data_type=self.data_type,
                            converted_name=f"{converted_prefix}moe/{i}/mlp/c_gate/weight:0",
                            reshape_fn=self.linear_weight_reshape,
                        ),
                        ConvertInfo(
                            param_names=[
                                f"{layer_prefix}block_sparse_moe.experts.{i}.w2.weight"
                            ],
                            data_type=self.data_type,
                            converted_name=f"{converted_prefix}moe/{i}/mlp/c_proj/weight:0",
                            reshape_fn=self.linear_weight_reshape,
                        ),
                        ConvertInfo(
                            param_names=[
                                f"{layer_prefix}block_sparse_moe.experts.{i}.w3.weight"
                            ],
                            data_type=self.data_type,
                            converted_name=f"{converted_prefix}moe/{i}/mlp/c_fc/weight:0",
                            reshape_fn=self.linear_weight_reshape,
                        ),
                    ]
                )
        return convert_info_list

    @property
    def model_type(self) -> str:
        """Model type."""
        return "mixtral"

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in Mixtral."""
        return cast(MixtralConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in Mixtral."""
        return cast(MixtralConfig, self.config).hidden_size

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of linear layer in Mixtral MoEs."""
        return cast(MixtralConfig, self.config).intermediate_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in Mixtral."""
        return cast(MixtralConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in Mixtral."""
        config = cast(MixtralConfig, self.config)
        if config.num_key_value_heads is None:
            return self.decoder_num_attention_heads
        return config.num_key_value_heads

    @property
    def attention_window_size(self) -> Optional[int]:
        """The size of sliding window attention in Mixtral."""
        return cast(MixtralConfig, self.config).sliding_window

    @property
    def num_experts(self) -> int:
        """The number of moe experts per transformer block in Mixtral."""
        return cast(MixtralConfig, self.config).num_local_experts

    @property
    def num_selected_moe_experts(self) -> int:
        """The number of selected moe experts per transformer block in Mixtral."""
        return cast(MixtralConfig, self.config).num_experts_per_tok
