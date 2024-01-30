# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Friendli Phi Checkpoint Converter."""


from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, cast

import torch
from transformers import PretrainedConfig  # type: ignore[import]

from friendli.errors import CheckpointConversionError, NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.base import (
    DECODER_PREFIX,
    SUPPORTED_GELU_FAMILY,
    DecoderOnlyConverter,
)
from friendli.modules.converter.interface import RotaryEmbeddingConversionInterface
from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.converter.utils import convert_to_gpt_j_params


class PhiMsftConfig(PretrainedConfig):
    """Phi msft configuration. Different from the HuggingFace PhiConfig."""

    model_type = "phi"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size: int = 50304,
        n_positions: int = 2048,
        n_embd: int = 1024,
        n_layer: int = 20,
        n_inner: Optional[int] = None,
        n_head: int = 16,
        n_head_kv: Optional[int] = None,
        rotary_dim: Optional[int] = 32,
        activation_function: Optional[str] = "gelu_new",
        flash_attn: bool = False,
        flash_rotary: bool = False,
        fused_dense: bool = False,
        attn_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        pad_vocab_size_multiple: int = 64,
        **kwargs,
    ) -> None:
        """Initalize the configuration for a phi-msft model."""
        self.vocab_size = int(
            math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        )
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.rotary_dim = min(rotary_dim, n_embd // n_head)  # type: ignore[type-var]
        self.activation_function = activation_function
        self.flash_attn = flash_attn
        self.flash_rotary = flash_rotary
        self.fused_dense = fused_dense
        self.attn_pdrop = attn_pdrop
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class PhiForCausalLMConverter(DecoderOnlyConverter):
    """PhiForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if Phi architectures' config can be converted to Friendli format."""
        super().check_config()
        try:
            if (
                cast(PhiMsftConfig, self.config).activation_function
                not in SUPPORTED_GELU_FAMILY
            ):
                raise NotSupportedCheckpointError(
                    invalid_option="'activation_function="
                    f"{cast(PhiMsftConfig, self.config).activation_function}'",
                    valid_options=SUPPORTED_GELU_FAMILY,
                )
            if cast(PhiMsftConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def qkv_weight_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """qkv_weight_reshape for Phi's attention layer."""
        assert len(params) == 1
        qkv_weight = params[0]

        q_size = self.decoder_num_attention_heads * self.decoder_head_size
        kv_size = self.decoder_num_kv_attention_heads * self.decoder_head_size
        q_weight, k_weight, v_weight = torch.split(
            qkv_weight, [q_size, kv_size, kv_size], dim=0
        )

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

    def qkv_bias_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """qkv_bias_reshape for Phi's attention layer."""
        assert len(params) == 1
        qkv_bias = params[0]

        q_size = self.decoder_num_attention_heads * self.decoder_head_size
        kv_size = self.decoder_num_kv_attention_heads * self.decoder_head_size

        q_bias, k_bias, v_bias = torch.split(
            qkv_bias, [q_size, kv_size, kv_size], dim=0
        )

        q_bias = q_bias.reshape(
            self.decoder_num_attention_heads, self.decoder_head_size
        )
        k_bias = k_bias.reshape(
            self.decoder_num_kv_attention_heads, self.decoder_head_size
        )

        q_bias = convert_to_gpt_j_params(q_bias, self.rotary_dim).flatten()
        k_bias = convert_to_gpt_j_params(k_bias, self.rotary_dim).flatten()

        qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
        return qkv_bias

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(PhiMsftConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The Phi model does not rely on "
            "absolute position embeddings, allowing you to choose any suitable value.",
            config.n_positions,
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
            "max_length": config.n_positions,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
            "rope_theta": self.rotary_emb_base,
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "phi"

    @property
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for transformer blocks in Phi."""
        convert_info_list = []
        for i in range(self.decoder_layer_num):
            layer_prefix = f"{self.decoder_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ln.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/gamma:0",
                        reshape_fn=self.ln_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}ln.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}ln_1/beta:0",
                        reshape_fn=self.ln_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.fc1.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.fc1.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_fc/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.fc2.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.fc2.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}mlp/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mixer.Wqkv.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/weight:0",
                        reshape_fn=self.qkv_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mixer.Wqkv.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_attn/bias:0",
                        reshape_fn=self.qkv_bias_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mixer.out_proj.weight"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/weight:0",
                        reshape_fn=self.linear_weight_reshape,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mixer.out_proj.bias"],
                        data_type=self.data_type,
                        converted_name=f"{converted_prefix}attn/c_proj/bias:0",
                        reshape_fn=self.linear_bias_reshape,
                    ),
                ]
            )
        return convert_info_list

    @property
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """The list of conversion informations for non-transformer blocks in Phi."""
        return [
            ConvertInfo(
                param_names=["transformer.embd.wte.weight"],
                data_type=self.data_type,
                converted_name="wte/weight:0",
                reshape_fn=self.token_embed_weight_reshape,
            ),
            ConvertInfo(
                param_names=["lm_head.ln.weight"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/gamma:0",
                reshape_fn=self.ln_weight_reshape,
            ),
            ConvertInfo(
                param_names=["lm_head.ln.bias"],
                data_type=self.data_type,
                converted_name=f"{DECODER_PREFIX}/ln_f/beta:0",
                reshape_fn=self.ln_bias_reshape,
            ),
            ConvertInfo(
                param_names=["lm_head.linear.weight"],
                data_type=self.data_type,
                converted_name="head_fc/weight:0",
                reshape_fn=self.head_weight_reshape,
            ),
            ConvertInfo(
                param_names=["lm_head.linear.bias"],
                data_type=self.data_type,
                converted_name="head_fc/bias:0",
                reshape_fn=self.head_weight_reshape,
            ),
        ]

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before Phi's transformer module number."""
        return "transformer.h."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in Phi."""
        return cast(PhiMsftConfig, self.config).n_layer

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in Phi."""
        return cast(PhiMsftConfig, self.config).n_embd

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in Phi."""
        return cast(PhiMsftConfig, self.config).n_head

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in Phi."""
        config = cast(PhiMsftConfig, self.config)
        if config.n_head_kv is not None:
            return config.n_head_kv
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head size of Phi."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def decoder_ff_intermediate_size(self) -> int:
        """The intermediate size of the linear layer in codegen MLP."""
        config = cast(PhiMsftConfig, self.config)
        if config.n_inner is None:
            return self.decoder_hidden_size * 4
        return config.n_inner

    @property
    def rotary_dim(self) -> int:
        """The rotary dim in Phi."""
        return cast(PhiMsftConfig, self.config).rotary_dim  # type: ignore[return-value]

    @property
    def rotary_emb_base(self) -> float:
        """The rotary emb base in Phi."""
        return 10000.0
