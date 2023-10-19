# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CodeGen Checkpoint Converter."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, cast

import numpy as np
import torch
from transformers import CodeGenConfig  # type: ignore[import]

from periflow.errors import CheckpointConversionError, NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.base import (
    DECODER_PREFIX,
    SUPPORTED_GELU_FAMILY,
    DecoderOnlyConverter,
)
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    get_tensor_from_state_dict,
    nontype_partial,
)


class CodegenForCausalLMConverter(DecoderOnlyConverter):
    """CodegenForCausalLM Architectures Converter Class."""

    def check_config(self) -> None:
        """Check if CodeGen architectures' config can be converted to PeriFlow format."""
        super().check_config()
        try:
            if (
                cast(CodeGenConfig, self.config).activation_function
                not in SUPPORTED_GELU_FAMILY
            ):
                raise NotSupportedCheckpointError(
                    invalid_option="'activation_function="
                    f"{cast(CodeGenConfig, self.config).activation_function}'",
                    valid_options=SUPPORTED_GELU_FAMILY,
                )
            if cast(CodeGenConfig, self.config).tie_word_embeddings:
                raise NotSupportedCheckpointError(
                    invalid_option="'tie_word_embeddings=True'",
                    valid_options=[False],
                )
            if cast(CodeGenConfig, self.config).layer_norm_epsilon != 1e-5:
                raise NotSupportedCheckpointError(
                    invalid_option="'layer_norm_epsilon="
                    f"{cast(CodeGenConfig, self.config).layer_norm_epsilon}'",
                    valid_options=[1e-5],
                )
        except AttributeError as exc:
            raise CheckpointConversionError(str(exc)) from exc

    def qkv_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """qkv_weight_convert for CodeGen's attention layer."""
        original_qkv_weight = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[0]
        )
        reshaped_qkv_weight = original_qkv_weight.reshape(
            (4, original_qkv_weight.size(0) // 4, original_qkv_weight.size(1))
        )
        q_weight, v_weight, k_weight = torch.split(
            reshaped_qkv_weight, reshaped_qkv_weight.size(1) // 3, dim=1
        )
        q_weight = q_weight.reshape((-1, q_weight.size(2)))
        k_weight = k_weight.reshape((-1, k_weight.size(2)))
        v_weight = v_weight.reshape((-1, v_weight.size(2)))

        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        qkv_weight = qkv_weight.transpose(0, 1)

        return convert_tensor_to_np_array(qkv_weight, self.data_type)

    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""
        config = cast(CodeGenConfig, self.config)

        logger.info(
            "The generated attributes set 'max_length' to %d, but you can change the "
            "'max_length' according to your needs. The CodeGen model does not rely on "
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
            "num_layers": self.decoder_layer_num,
            "max_length": config.n_positions,
            "vocab_size": config.vocab_size,
            "eos_token": eos_token_id if eos_token_id is not None else "FILL ME",
        }
        return attr

    @property
    def model_type(self) -> str:
        """Model type."""
        return "gpt-j"

    @property
    def non_transformer_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """The convert_dict for non-transformer blocks in CodeGen."""
        return {
            "wte/weight:0": nontype_partial(
                self.token_embed_weight_convert,
                per_layer_postfixes=["transformer.wte.weight"],
            ),
            DECODER_PREFIX
            + "/ln_f/gamma:0": nontype_partial(
                self.ln_weight_convert, per_layer_postfixes=["transformer.ln_f.weight"]
            ),
            DECODER_PREFIX
            + "/ln_f/beta:0": nontype_partial(
                self.ln_bias_convert, per_layer_postfixes=["transformer.ln_f.bias"]
            ),
            "head_fc/weight:0": nontype_partial(
                self.head_weight_convert, per_layer_postfixes=["lm_head.weight"]
            ),
            "head_fc/bias:0": nontype_partial(
                self.linear_bias_convert, per_layer_postfixes=["lm_head.bias"]
            ),
        }

    @property
    def decoder_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """The convert_dict for transformer blocks in CodeGen."""
        return {
            "ln_1/gamma:0": nontype_partial(
                self.ln_weight_convert,
                per_layer_postfixes=[".ln_1.weight"],
            ),
            "ln_1/beta:0": nontype_partial(
                self.ln_bias_convert,
                per_layer_postfixes=[".ln_1.bias"],
            ),
            "mlp/c_fc/bias:0": nontype_partial(
                self.linear_bias_convert,
                per_layer_postfixes=[".mlp.fc_in.bias"],
            ),
            "mlp/c_proj/bias:0": nontype_partial(
                self.linear_bias_convert,
                per_layer_postfixes=[".mlp.fc_out.bias"],
            ),
            "attn/c_attn/weight:0": nontype_partial(
                self.qkv_weight_convert,
                per_layer_postfixes=[".attn.qkv_proj.weight"],
            ),
            "attn/c_proj/weight:0": nontype_partial(
                self.linear_weight_convert,
                per_layer_postfixes=[".attn.out_proj.weight"],
            ),
            "mlp/c_fc/weight:0": nontype_partial(
                self.linear_weight_convert,
                per_layer_postfixes=[".mlp.fc_in.weight"],
            ),
            "mlp/c_proj/weight:0": nontype_partial(
                self.linear_weight_convert,
                per_layer_postfixes=[".mlp.fc_out.weight"],
            ),
        }

    @property
    def decoder_layer_prefix(self) -> str:
        """The layer name prefix used before CodeGen's transformer block number."""
        return "transformer.h."

    @property
    def decoder_layer_num(self) -> int:
        """The number of decoder layers in CodeGen."""
        return cast(CodeGenConfig, self.config).num_hidden_layers

    @property
    def decoder_hidden_size(self) -> int:
        """The hidden size in CodeGen."""
        return cast(CodeGenConfig, self.config).hidden_size

    @property
    def decoder_num_attention_heads(self) -> int:
        """The number of attention heads in CodeGen."""
        return cast(CodeGenConfig, self.config).num_attention_heads

    @property
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads in the codegen."""
        return self.decoder_num_attention_heads

    @property
    def decoder_head_size(self) -> int:
        """The head siez of CodeGen."""
        return self.decoder_hidden_size // self.decoder_num_attention_heads

    @property
    def rotary_dim(self) -> int:
        """The rotary dim in CodeGen."""
        return cast(CodeGenConfig, self.config).rotary_dim
