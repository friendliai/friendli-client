# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint Converter Interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import h5py  # type: ignore[import]
import numpy as np
import torch
from tqdm.std import tqdm as std_tqdm

from periflow.modules.converter.utils import get_tensor_from_state_dict, nontype_partial

ENCODER_PREFIX = "encoder"
DECODER_PREFIX = "decoder"


class ConversionInterface(ABC):
    """Interface get information for converting common layers."""

    @property
    @abstractmethod
    def non_transformer_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for layers that do not belong to transformer blocks."""

    def convert_non_transformer_layers(
        self, state_dict: Dict[str, torch.Tensor], out_f: h5py.File, pbar: std_tqdm
    ) -> None:
        """Return the number of transformer blocks converted in the decoder."""
        for (
            converted_layer_name,
            convert_fn,
        ) in self.non_transformer_convert_dict.items():
            converted_np_array = convert_fn(state_dict, "")
            out_f[converted_layer_name] = converted_np_array
            pbar.update()


class DecoderConversionInterface(ABC):
    """Interface get information for converting decoder layers."""

    @property
    @abstractmethod
    def decoder_layer_prefix(self) -> str:
        """Return the layer name prefix used before the decoder's transformer block number."""

    @property
    @abstractmethod
    def decoder_layer_num(self) -> int:
        """Return the number of transformer blocks in the decoder."""

    @property
    @abstractmethod
    def decoder_hidden_size(self) -> int:
        """Return the hidden size of the decoder."""

    @property
    @abstractmethod
    def decoder_num_kv_attention_heads(self) -> int:
        """The number of key-value attention heads."""

    @property
    @abstractmethod
    def decoder_num_attention_heads(self) -> int:
        """Return the number of attention heads in the decoder."""

    @property
    @abstractmethod
    def decoder_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for transformer blocks in the decoder."""

    @property
    @abstractmethod
    def decoder_head_size(self) -> int:
        """Return the head size of the decoder."""

    def convert_decoder_layers(
        self, state_dict: Dict[str, torch.Tensor], out_f: h5py.File, pbar: std_tqdm
    ) -> None:
        """Return the number of transformer blocks converted in the decoder."""
        out_f.create_group(DECODER_PREFIX)
        for i in range(self.decoder_layer_num):
            layer = self.decoder_layer_prefix + f"{i}"
            per_layer_out_ckpt = out_f[DECODER_PREFIX].create_group(f"h_._{i}")

            for converted_layer_name, convert_fn in self.decoder_convert_dict.items():
                converted_np_array = convert_fn(state_dict, layer)
                per_layer_out_ckpt[converted_layer_name] = converted_np_array
                pbar.update()


class EncoderConversionInterface(ABC):
    """Interface get information for converting encoder layers."""

    @property
    @abstractmethod
    def encoder_layer_prefix(self) -> str:
        """Return the layer name prefix used before the encoder's transformer block number."""

    @property
    @abstractmethod
    def encoder_layer_num(self) -> int:
        """Return the number of transformer blocks in the encoder."""

    @property
    @abstractmethod
    def encoder_hidden_size(self) -> int:
        """Return the hidden size of the encoder."""

    @property
    @abstractmethod
    def encoder_num_attention_heads(self) -> int:
        """Return the number of attention heads in the encoder."""

    @property
    @abstractmethod
    def encoder_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for transformer blocks in the encoder."""

    @property
    @abstractmethod
    def encoder_head_size(self) -> int:
        """Return the head size of the encoder."""

    def convert_encoder_layers(
        self, state_dict: Dict[str, torch.Tensor], out_f: h5py.File, pbar: std_tqdm
    ) -> None:
        """Return the number of transformer blocks converted in the encoder."""
        out_f.create_group(ENCODER_PREFIX)
        for i in range(self.encoder_layer_num):
            layer = self.encoder_layer_prefix + f"{i}"
            per_layer_out_ckpt = out_f[ENCODER_PREFIX].create_group(f"h_._{i}")

            for converted_layer_name, convert_fn in self.encoder_convert_dict.items():
                converted_np_array = convert_fn(state_dict, layer)
                per_layer_out_ckpt[converted_layer_name] = converted_np_array
                pbar.update()


class QuantConversionInterface(ABC):
    """Interface get information for converting quantized layers."""

    @property
    def quantized_param_names(self) -> List[str]:
        """Return the parameter names of quantized layers."""
        return [
            "attn/c_attn/weight:0",
            "attn/c_proj/weight:0",
            "mlp/c_fc/weight:0",
            "mlp/c_proj/weight:0",
        ]

    @property
    def quantized_layer_prefix(self) -> str:
        """The layer name prefix used before transformer block number."""
        return self.decoder_layer_prefix  # type: ignore[attr-defined] # pylint: disable=no-member

    @property
    def quantized_layer_num(self) -> int:
        """Return the number of quantized transformer blocks ."""
        return self.decoder_layer_num  # type: ignore[attr-defined] # pylint: disable=no-member

    @property
    def quantized_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for quantized layers."""
        return {
            "attn/c_attn/q_weight_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".q.weight_scale"]
            ),
            "attn/c_attn/k_weight_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".k.weight_scale"]
            ),
            "attn/c_attn/v_weight_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".v.weight_scale"]
            ),
            "attn/c_attn/q_out_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".q.out_scale"]
            ),
            "attn/c_attn/k_out_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".k.out_scale"]
            ),
            "attn/c_attn/v_out_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".v.out_scale"]
            ),
            "attn/c_attn/in_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".q.in_scale"]
            ),
            "attn/c_proj/weight_scale:0": nontype_partial(
                self.scale_convert,
                per_layer_postfixes=[".attn_fc.weight_scale"],
            ),
            "attn/c_proj/out_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".attn_fc.out_scale"]
            ),
            "attn/c_proj/in_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".attn_fc.in_scale"]
            ),
            "mlp/c_fc/weight_scale:0": nontype_partial(
                self.scale_convert, per_layer_postfixes=[".ff1.weight_scale"]
            ),
            "mlp/c_fc/out_scale:0": nontype_partial(
                self.scale_convert,
                per_layer_postfixes=[".ff1.out_scale"],
            ),
            "mlp/c_fc/in_scale:0": nontype_partial(
                self.scale_convert,
                per_layer_postfixes=[".ff1.in_scale"],
            ),
            "mlp/c_proj/weight_scale:0": nontype_partial(
                self.scale_convert,
                per_layer_postfixes=[".ff2.weight_scale"],
            ),
            "mlp/c_proj/out_scale:0": nontype_partial(
                self.scale_convert,
                per_layer_postfixes=[".ff2.out_scale"],
            ),
            "mlp/c_proj/in_scale:0": nontype_partial(
                self.scale_convert,
                per_layer_postfixes=[".ff2.in_scale"],
            ),
            "attn/c_attn/int8_weight:0": nontype_partial(
                self.quantized_qkv_weight_convert,
                per_layer_postfixes=[
                    ".q.int8_weight",
                    ".k.int8_weight",
                    ".v.int8_weight",
                ],
            ),
            "attn/c_proj/int8_weight:0": nontype_partial(
                self.quantized_linear_weight_convert,
                per_layer_postfixes=[".attn_fc.int8_weight"],
            ),
            "mlp/c_fc/int8_weight:0": nontype_partial(
                self.quantized_linear_weight_convert,
                per_layer_postfixes=[".ff1.int8_weight"],
            ),
            "mlp/c_proj/int8_weight:0": nontype_partial(
                self.quantized_linear_weight_convert,
                per_layer_postfixes=[".ff2.int8_weight"],
            ),
        }

    def scale_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert scale of quantized layers."""
        assert len(per_layer_postfixes) == 1
        converted_np_array = np.array(
            state_dict[prefix + per_layer_postfixes[0]], dtype=np.float32
        )
        return converted_np_array

    def quantized_qkv_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert weight of quantized qkv layers."""
        assert len(per_layer_postfixes) == 3
        qkv_weight = torch.concat(
            [
                get_tensor_from_state_dict(state_dict, layer + postfix)
                for postfix in per_layer_postfixes
            ],
            dim=0,
        )  # [OutDim, InDim]
        return qkv_weight.cpu().detach().numpy()

    def quantized_linear_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert weight of quantized linear layers."""
        assert len(per_layer_postfixes) == 1
        linear_int8_weight = get_tensor_from_state_dict(
            state_dict, layer + per_layer_postfixes[0]
        )  # [OutDim, InDim]
        return linear_int8_weight.cpu().detach().numpy()

    def convert_quantized_layers(
        self, state_dict: Dict[str, torch.Tensor], out_f: h5py.File, pbar: std_tqdm
    ) -> None:
        """Return the number of transformer blocks converted in the decoder."""
        for i in range(self.quantized_layer_num):
            layer = f"{i}"
            per_layer_out_ckpt = out_f[DECODER_PREFIX + f"/h_._{i}"]

            for converted_layer_name, convert_fn in self.quantized_convert_dict.items():
                converted_np_array = convert_fn(state_dict, layer)
                per_layer_out_ckpt[converted_layer_name] = converted_np_array
                pbar.update()
