# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow GPTNeoXForCausalLM QuantizerHook."""

# mypy: ignore-errors

from __future__ import annotations

from typing import Iterator, List, Tuple, Type

import torch

from periflow.enums import CheckpointDataType
from periflow.modules.converter.base import DECODER_PREFIX
from periflow.modules.converter.schema import ConvertInfo
from periflow.modules.converter.utils import convert_to_gpt_j_params
from periflow.modules.quantizer.awq.base import AWQHook
from periflow.modules.quantizer.schema.data import ModuleName, QuantInput, TFQuantInputs
from periflow.modules.quantizer.utils import scale_convert


class AWQGPTNeoXHook(AWQHook):
    """AWQ Hook for GPTNeoXForCausalLM."""

    def __init__(self, quant_config, converter):
        """Initialize AWQGPTNeoXHook."""
        super().__init__(quant_config, converter)
        config = converter.config
        self.data_type = converter.data_type
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_dim = int(self.head_size * config.rotary_pct)
        assert config.use_parallel_residual == True

    def add_pre_scaler(self, model: torch.nn.Module) -> torch.nn.Module:
        """Adds scaler to GPTNeoXForCausalLM."""
        for tf_block in self.get_tf_blocks(model):
            attn_fc_scaler = self._register_pre_scaler(
                tf_block.attention.dense,
            )
            tf_block.attention.add_module("scaler", attn_fc_scaler)
            ff2_scaler = self._register_pre_scaler(tf_block.mlp.dense_4h_to_h)
            tf_block.mlp.add_module("scaler", ff2_scaler)
        return model

    def get_inspect_module_types(
        self, block: torch.nn.Module
    ) -> Tuple[Type[torch.nn.Module], ...]:
        """Returns the type of linear layer (etc. qkv, linear layer) in transformer block."""
        return (type(block.attention), type(block.mlp))

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
            [block.input_layernorm],
            [("attention.query_key_value", block.attention.query_key_value)],
            block.attention,
            "attention",
        )
        # attn out proj
        yield (
            [block.attention.scaler],
            [("attention.dense", block.attention.dense)],
            block.attention.dense,
            "attention.dense",
        )
        # ff1
        yield (
            [block.post_attention_layernorm],
            [("mlp.dense_h_to_4h", block.mlp.dense_h_to_4h)],
            block.mlp,
            "mlp",
        )
        # ff2
        yield (
            [block.mlp.scaler],
            [("mlp.dense_4h_to_h", block.mlp.dense_4h_to_h)],
            block.mlp.dense_4h_to_h,
            "mlp.dense_4h_to_h",
        )

    def reshape_qkv_weight(
        self, attn_layer: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape GPTNeoXForCausalLM's qkv weight."""
        qkv_weight = attn_layer.query_key_value.weight
        qkv_weight = qkv_weight.reshape(
            self.num_attention_heads,
            3,
            self.head_size,
            self.hidden_size,
        )

        q_weight = qkv_weight[:, 0].reshape(
            self.num_attention_heads,
            self.head_size,
            self.hidden_size,
        )
        k_weight = qkv_weight[:, 1].reshape(
            self.num_attention_heads,
            self.head_size,
            self.hidden_size,
        )
        v_weight = qkv_weight[:, 2].reshape(
            self.num_attention_heads * self.head_size,
            self.hidden_size,
        )

        q_weight = convert_to_gpt_j_params(param=q_weight, rotary_dim=self.rotary_dim)
        k_weight = convert_to_gpt_j_params(param=k_weight, rotary_dim=self.rotary_dim)
        q_weight = q_weight.reshape(
            self.num_attention_heads * self.head_size,
            self.hidden_size,
        )
        k_weight = k_weight.reshape(
            self.num_attention_heads * self.head_size,
            self.hidden_size,
        )
        return q_weight, k_weight, v_weight

    def iter_quant_inputs(self, model: torch.nn.Module) -> Iterator[TFQuantInputs]:
        """Returns the layers which should be quantized in transformer block of GPTNeoXForCausalLM."""
        for index, decoder_layer in enumerate(
            self.get_tf_blocks(model)  # type: ignore[union-attr, arg-type]
        ):
            q_weight, k_weight, v_weight = self.reshape_qkv_weight(
                decoder_layer.attention
            )
            qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            attn_weight_outdim = qkv_weight.size(0)  # OutDim

            yield TFQuantInputs(
                layer_index=index,
                parent_module=decoder_layer,
                q=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.attention.query_key_value",
                    0,
                    attn_weight_outdim // 3,
                ),
                k=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.attention.query_key_value",
                    attn_weight_outdim // 3,
                    attn_weight_outdim // 3 * 2,
                ),
                v=QuantInput(
                    qkv_weight,
                    f"{self.quantized_layer_prefix}{index}.attention.query_key_value",
                    attn_weight_outdim // 3 * 2,
                    attn_weight_outdim,
                ),
                attn_fc=QuantInput(
                    decoder_layer.attention.dense.weight,
                    f"{self.quantized_layer_prefix}{index}.attention.dense",
                    None,
                    None,
                ),
                ff1=QuantInput(
                    decoder_layer.mlp.dense_h_to_4h.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.dense_h_to_4h",
                    None,
                    None,
                ),
                ff2=QuantInput(
                    decoder_layer.mlp.dense_4h_to_h.weight,
                    f"{self.quantized_layer_prefix}{index}.mlp.dense_4h_to_h",
                    None,
                    None,
                ),
            )

    def get_linear_layer_types(self) -> Tuple[Type[torch.nn.Module]]:
        """Returns the linear layer types in GPTNeoXForCausalLM."""
        return (torch.nn.Linear,)

    def get_tf_blocks(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Returns the transformer blocks in GPTNeoXForCausalLM."""
        return model.gpt_neox.layers  # type: ignore

    @property
    def modified_layers_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for modified layers."""
        convert_info_list = []
        for i in range(self.converter.decoder_layer_num):
            layer_prefix = f"{self.quantized_layer_prefix}{i}."
            converted_prefix = f"{DECODER_PREFIX}/h_._{i}/"
            convert_info_list.extend(
                [
                    ConvertInfo(
                        param_names=[f"{layer_prefix}attention.scaler.scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}attn/c_proj/awq/pre_scale:0",
                        convert_fn=scale_convert,
                    ),
                    ConvertInfo(
                        param_names=[f"{layer_prefix}mlp.scaler.scale"],
                        data_type=CheckpointDataType.FP32,
                        converted_name=f"{converted_prefix}mlp/c_proj/awq/pre_scale:0",
                        convert_fn=scale_convert,
                    ),
                ]
            )
        return convert_info_list

    @property
    def avoid_clipping_layer_names(self) -> List[str]:
        """Returns the layer names which should be avoided for AWQ clipping."""
        return ["query_key_value"]
