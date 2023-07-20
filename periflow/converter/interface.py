# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint Converter Interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import h5py  # type: ignore[import]
import torch
from tqdm.std import tqdm as std_tqdm

ENCODER_PREFIX = "encoder"
DECODER_PREFIX = "decoder"


class ConversionInterface(ABC):
    """Interface get information for converting common layers."""

    @property
    @abstractmethod
    def non_transformer_convert_dict(self) -> Dict[str, Any]:
        """Return the convert_dict for layers that do not belong to Transformer layers."""

    def convert_non_transformer_layers(
        self, state_dict: Dict[str, torch.Tensor], out_f: h5py.File, pbar: std_tqdm
    ) -> None:
        """Return the number of transformer layers converted in the decoder."""
        for converted_layer_name, (
            convert_fn,
            per_layer_postfixes,
        ) in self.non_transformer_convert_dict.items():
            converted_tensor = convert_fn(state_dict, "", per_layer_postfixes)
            out_f[converted_layer_name] = converted_tensor
            pbar.update()


class DecoderConversionInterface(ABC):
    """Mixin class get information for converting decoder layers."""

    @property
    @abstractmethod
    def decoder_layer_prefix(self) -> str:
        """Return the layer name prefix used before the decoder's transformer layer number."""

    @property
    @abstractmethod
    def decoder_layer_num(self) -> int:
        """Return the number of transformer layers in the decoder."""

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
    def decoder_convert_dict(self) -> Dict[str, Any]:
        """Return the convert_dict for transformer layers in the decoder."""

    @property
    @abstractmethod
    def decoder_head_size(self) -> int:
        """Return the head size of the decoder."""

    def convert_decoder_layers(
        self, state_dict: Dict[str, torch.Tensor], out_f: h5py.File, pbar: std_tqdm
    ) -> None:
        """Return the number of transformer layers converted in the decoder."""
        out_f.create_group(DECODER_PREFIX)
        for i in range(self.decoder_layer_num):
            layer = self.decoder_layer_prefix + f"{i}"
            per_layer_out_ckpt = out_f[DECODER_PREFIX].create_group(f"h_._{i}")

            for converted_layer_name, (
                convert_fn,
                per_layer_postfixes,
            ) in self.decoder_convert_dict.items():
                converted_tensor = convert_fn(state_dict, layer, per_layer_postfixes)
                per_layer_out_ckpt[converted_layer_name] = converted_tensor
                pbar.update()


class EncoderConversionInterface(ABC):
    """Mixin class get information for converting encoder layers."""

    @property
    @abstractmethod
    def encoder_layer_prefix(self) -> str:
        """Return the layer name prefix used before the encoder's transformer layer number."""

    @property
    @abstractmethod
    def encoder_layer_num(self) -> int:
        """Return the number of transformer layers in the encoder."""

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
    def encoder_convert_dict(self) -> Dict[str, Any]:
        """Return the convert_dict for transformer layers in the encoder."""

    @property
    @abstractmethod
    def encoder_head_size(self) -> int:
        """Return the head size of the encoder."""

    def convert_encoder_layers(
        self, state_dict: Dict[str, torch.Tensor], out_f: h5py.File, pbar: std_tqdm
    ) -> None:
        """Return the number of transformer layers converted in the encoder."""
        out_f.create_group(ENCODER_PREFIX)
        for i in range(self.encoder_layer_num):
            layer = self.encoder_layer_prefix + f"{i}"
            per_layer_out_ckpt = out_f[ENCODER_PREFIX].create_group(f"h_._{i}")

            for converted_layer_name, (
                convert_fn,
                per_layer_postfixes,
            ) in self.encoder_convert_dict.items():
                converted_tensor = convert_fn(
                    state_dict=state_dict,
                    layer=layer,
                    per_layer_postfixes=per_layer_postfixes,
                )
                per_layer_out_ckpt[converted_layer_name] = converted_tensor
                pbar.update()
