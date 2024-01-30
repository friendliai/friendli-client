# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli MPT Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, List, cast

from transformers import (  # type: ignore[import]
    GenerationConfig,
    MptConfig,
    PretrainedConfig,
)

from friendli.enums import CheckpointDataType  # type: ignore[import]
from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import (
    DECODER_PREFIX,
    DecoderOnlyConverter,
    DecoderOnlyLoraConverter,
)
from friendli.modules.converter.schema import ConvertInfo


def safe_attn_config_get(attn_config: Dict[str, Any], key: str) -> Any:
    """Safe getter from MptAttentionConfig.

    This function is a temporary function because MptAttentionConfig
    is not supported `attn_type="grouped_query_attention"` yet.
    """
    if key not in attn_config:
        raise CheckpointConversionError(
            f"{key} does not exist in MptAttentionConfig {attn_config}"
        )

    return attn_config[key]


class MptForCausalLMLoraConverter(DecoderOnlyLoraConverter):
    """MptForCausalLM LoRA Converter Class."""

    @property
    def adapter_target_modules(self) -> List[str]:
        """Return the target modules that LoRA applies to."""
        return ["merged-qkv"]

    @property
    def adapter_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for LoRA adapter modules in Mpt."""
        convert_info_list = []
        for i in range(self.converter.decoder_layer_num):
            layer_prefix = f"{self.converter.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn.Wqkv.lora_A.default.weight"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/lora/lora_A/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn.Wqkv.lora_B.default.weight"],
                        data_type=self.converter.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/lora/lora_B/weight:0",
                        reshape_fn=self.lora_weight_reshape,
                    ),
                ]
            )
        return convert_info_list


class MPTForCausalLMConverter(DecoderOnlyConverter):
    """MPTForCausalLM Architectures Converter Class."""

    def __init__(
        self,
        config: PretrainedConfig,
        generation_config: GenerationConfig | None,
        data_type: CheckpointDataType,
    ) -> None:
        """Initialize MPTForCausalLMConverter."""
        super().__init__(config, generation_config, data_type)
        attn_config = cast(MptConfig, config).attn_config
        if isinstance(attn_config, PretrainedConfig):
            attn_config = attn_config.to_dict()  # type: ignore
        self.attn_config = attn_config

    def check_config(self) -> None:
        """Check if MPT architectures' config can be converted to Friendli format."""
        super().check_config()

        if not safe_attn_config_get(self.attn_config, "alibi"):
            raise NotSupportedCheckpointError(
                invalid_option=f"'alibi={safe_attn_config_get(self.attn_config, 'alibi')}'",
                valid_options=[True],
            )

        if safe_attn_config_get(self.attn_config, "alibi_bias_max") != 8:
            raise NotSupportedCheckpointError(
                invalid_option=f"'alibi={safe_attn_config_get(self.attn_config, 'alibi_bias_max')}'",
                valid_options=[8],
            )

        if safe_attn_config_get(self.attn_config, "attn_type") != "multihead_attention":
            if (
                safe_attn_config_get(self.attn_config, "attn_type")
                == "grouped_query_attention"
            ):
                raise CheckpointConversionError(
                    msg="MptAttentionConfig does not support `attn_type=`grouped_query_attention`` yet (as of transformers==4.35.2).",
                )
            raise NotSupportedCheckpointError(
                invalid_option=f"'attn_type={safe_attn_config_get(self.attn_config, 'attn_type')}'",
                valid_options=["multihead_attention"],
            )

        if safe_attn_config_get(self.attn_config, "prefix_lm"):
            raise NotSupportedCheckpointError(
                invalid_option=f"'prefix_lm={safe_attn_config_get(self.attn_config, 'prefix_lm')}'",
                valid_options=[False],
            )

        if safe_attn_config_get(self.attn_config, "qk_ln"):
            raise NotSupportedCheckpointError(
                invalid_option=f"'qk_ln={safe_attn_config_get(self.attn_config, 'qk_ln')}'",
                valid_options=[False],
            )

        if safe_attn_config_get(self.attn_config, "softmax_scale") is not None:
            raise NotSupportedCheckpointError(
                invalid_option=f"'softmax_scale={safe_attn_config_get(self.attn_config, 'softmax_scale')}'",
                valid_options=[None],
            )

        if cast(MptConfig, self.config).expansion_ratio != 4:
            raise NotSupportedCheckpointError(
                invalid_option=(
                    f"'expansion_ratio={cast(MptConfig, self.config).expansion_ratio}'"
                ),
                valid_options=[4],
            )

        if not cast(MptConfig, self.config).no_bias:
            raise NotSupportedCheckpointError(
                invalid_option=f"'no_bias={cast(MptConfig, self.config).no_bias}'",
                valid_options=[True],
            )

        if cast(MptConfig, self.config).logit_scale is not None:
            raise NotSupportedCheckpointError(
                invalid_option=(
                    f"'logit_scale={cast(MptConfig, self.config).logit_scale}'"
                ),
                valid_options=[None],
            )

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in MPT."""
        convert_info_list = []
        for i in range(self.decoder_layer_num):
            layer_prefix = f"{self.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}norm_1.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}norm_2.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_2/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn.Wqkv.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/weight:0",
                        reshape_fn=self.qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attn.out_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ffn.up_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ffn.down_proj.weight"],
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
        """The list of conversion informations for non-transformer blocks in MPT."""
        return [
            ConvertInfo(
                param_names=["transformer.wte.weight"],
                data_type=self.data_type,
                converted_name="wte/weight:0",
                reshape_fn=self.token_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["transformer.norm_f.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/gamma:0",
                reshape_fn=self.ln_weight_reshape,
            ),
        ]

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The MPT model does not rely on "
            "absolute position embeddings, allowing you to choose any suitable value.",
            cast(MptConfig, self.config).max_seq_len,
        )

        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.decoder_head_size,
            "num_heads": self.decoder_num_attention_heads,
            "num_kv_heads": self.decoder_num_kv_attention_heads,
            "num_layers": self.decoder_layer_num,
            "max_length": cast(MptConfig, self.config).max_seq_len,
            "vocab_size": cast(MptConfig, self.config).vocab_size,
            "clip_qkv": safe_attn_config_get(self.attn_config, "clip_qkv") or 0.0,
            "eos_token": self.get_eos_token_id() or "FILL ME",
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "mpt"

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before the MPT's transformer block number."""
        return "transformer.blocks."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in MPT."""
        return cast(MptConfig, self.config).n_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in MPT."""
        return cast(MptConfig, self.config).d_model

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in MPT."""
        return cast(MptConfig, self.config).n_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in MPT."""
        if "kv_n_heads" in self.attn_config:
            return self.attn_config["kv_n_heads"]
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of MPT."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in MPT MLP."""
        return self.decoder_hidden_size * 4
