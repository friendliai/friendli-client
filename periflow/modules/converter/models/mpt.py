# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow MPT Checkpoint Converter."""

from __future__ import annotations

from typing import Any, Dict, Union

from transformers import PretrainedConfig  # type: ignore[import]

from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import DecoderOnlyConverter
from periflow.modules.converter.interface import DECODER_PREFIX


def safe_config_get(config: Union[Dict[str, Any], PretrainedConfig], key: str) -> Any:
    """Safe getter from config.

    This function is a temporary function because MPT is not merged into Hugging Face's upstream yet
    (i.e., we cannot do `from transformers import MPTConfig`)
    """
    if isinstance(config, PretrainedConfig):
        config = config.to_dict()  # type: ignore

    if key not in config:
        raise CheckpointConversionError(f"{key} does not exist in the config")

    return config[key]


class MPTForCausalLMConverter(DecoderOnlyConverter):
    """MPTForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if MPT architectures' config can be converted to PeriFlow format."""
        super().check_config()
        attn_config = safe_config_get(self.config, "attn_config")

        if not safe_config_get(attn_config, "alibi"):
            raise NotSupportedCheckpointError(
                invalid_option=f"'alibi={safe_config_get(attn_config, 'alibi')}'",
                valid_options=[True],
            )

        if safe_config_get(attn_config, "alibi_bias_max") != 8:
            raise NotSupportedCheckpointError(
                invalid_option=f"'alibi={safe_config_get(attn_config, 'alibi_bias_max')}'",
                valid_options=[8],
            )

        if safe_config_get(attn_config, "attn_type") != "multihead_attention":
            raise NotSupportedCheckpointError(
                invalid_option=f"'attn_type={safe_config_get(attn_config, 'attn_type')}'",
                valid_options=["multihead_attention"],
            )

        if safe_config_get(attn_config, "prefix_lm"):
            raise NotSupportedCheckpointError(
                invalid_option=f"'prefix_lm={safe_config_get(attn_config, 'prefix_lm')}'",
                valid_options=[False],
            )

        if safe_config_get(attn_config, "qk_ln"):
            raise NotSupportedCheckpointError(
                invalid_option=f"'qk_ln={safe_config_get(attn_config, 'qk_ln')}'",
                valid_options=[False],
            )

        if safe_config_get(attn_config, "softmax_scale") is not None:
            raise NotSupportedCheckpointError(
                invalid_option=f"'softmax_scale={safe_config_get(attn_config, 'softmax_scale')}'",
                valid_options=[None],
            )

        if safe_config_get(self.config, "expansion_ratio") != 4:
            raise NotSupportedCheckpointError(
                invalid_option=(
                    f"'expansion_ratio={safe_config_get(self.config, 'expansion_ratio')}'"
                ),
                valid_options=[4],
            )

        if not safe_config_get(self.config, "no_bias"):
            raise NotSupportedCheckpointError(
                invalid_option=f"'no_bias={safe_config_get(self.config, 'no_bias')}'",
                valid_options=[True],
            )

        if safe_config_get(self.config, "logit_scale") is not None:
            raise NotSupportedCheckpointError(
                invalid_option=(
                    f"'logit_scale={safe_config_get(self.config, 'logit_scale')}'"
                ),
                valid_options=[None],
            )

    @property
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for transformer layers in MPT."""
        return {
            "ln_1/gamma:0": (
                self.ln_weight_convert,
                [".norm_1.weight"],
            ),
            "attn/c_attn/weight:0": (
                self.qkv_weight_convert,
                [".attn.Wqkv.weight"],
            ),
            "attn/c_proj/weight:0": (
                self.linear_weight_convert,
                [".attn.out_proj.weight"],
            ),
            "ln_2/gamma:0": (
                self.ln_weight_convert,
                [".norm_2.weight"],
            ),
            "mlp/c_fc/weight:0": (
                self.linear_weight_convert,
                [".ffn.up_proj.weight"],
            ),
            "mlp/c_proj/weight:0": (
                self.linear_weight_convert,
                [".ffn.down_proj.weight"],
            ),
        }

    @property
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """The convert_dict for non-transformer layers in MPT."""
        return {
            "wte/weight:0": (
                self.token_embed_weight_convert,
                ["transformer.wte.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/gamma:0": (
                self.ln_weight_convert,
                ["transformer.norm_f.weight"],
            ),
        }

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The MPT model does not rely on "
            "absolute position embeddings, allowing you to choose any suitable value.",
            safe_config_get(self.config, "max_seq_len"),
        )

        attn_config = safe_config_get(self.config, "attn_config")

        attr = {
            "model_type": self.model_type,
            "dtype": self.data_type.value,
            "head_size": self.decoder_head_size,
            "num_heads": self.decoder_num_attention_heads,
            "num_layers": self.decoder_layer_num,
            "max_length": safe_config_get(self.config, "max_seq_len"),
            "vocab_size": safe_config_get(self.config, "vocab_size"),
            "clip_qkv": safe_config_get(attn_config, "clip_qkv") or 0.0,
            "eos_token": self.get_eos_token_id() or "FILL ME",
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "mpt"

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before the MPT's transformer layer number."""
        return "transformer.blocks."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in Falcon."""
        return safe_config_get(self.config, "n_layers")

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in Falcon."""
        return safe_config_get(self.config, "d_model")

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in Falcon."""
        return safe_config_get(self.config, "n_heads")

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in Falcon."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of Falcon."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads
