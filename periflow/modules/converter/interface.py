# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint Converter Interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import h5py  # type: ignore[import]
import numpy as np
import torch
from tqdm.std import tqdm as std_tqdm


class ModelConversionInterface(ABC):
    """Interface get information for converting models."""

    @abstractmethod
    def get_convert_dict(
        self,
    ) -> Dict[str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]]:
        """Return the convert_dict for the model.

        ### convert_dict format
        convert_dict = {
            "block-type": {
                "converted_layer_name": convert_fn,
            },
        }
        """

    @abstractmethod
    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""

    @abstractmethod
    def convert(
        self,
        model: torch.nn.Module,
        output_path: str,
        state_dict: Dict[str, torch.Tensor],
        convert_dict: Dict[
            str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]
        ],
    ) -> None:
        """Convert Huggingface Model to PeriFlow format(.h5).

        Args:
            model (torch.nn.Module): Huggingface model.
            output_path (str): Path to save the converted checkpoint.
            state_dict (Dict[str, torch.Tensor]):
                Dictionary of mapping of tensor name to tensor
            convert_dict (Dict[Callable[[Dict[str, torch.Tensor], str], np.ndarray]]):
                Dictionary of mapping converted params name to conversion functions.

        """

    @abstractmethod
    def check_config(self) -> None:
        """Check if the model is convertable."""


class NonTFBlockConversionInterface(ABC):
    """Interface get information for converting common layers."""

    @property
    @abstractmethod
    def non_transformer_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for layers that do not belong to transformer blocks."""

    def convert_non_transformer_layers(
        self,
        state_dict: Dict[str, torch.Tensor],
        convert_dict: Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]],
        out_f: h5py.Group,
        pbar: std_tqdm,
    ) -> None:
        """Return the number of transformer blocks converted in the decoder."""
        for (
            converted_layer_name,
            convert_fn,
        ) in convert_dict.items():
            converted_np_array = convert_fn(state_dict, "")
            out_f[converted_layer_name] = converted_np_array
            pbar.update()


class DecoderTFBlockConversionInterface(ABC):
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
        self,
        state_dict: Dict[str, torch.Tensor],
        convert_dict: Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]],
        out_f: h5py.Group,
        pbar: std_tqdm,
    ) -> None:
        """Return the number of transformer blocks converted in the decoder."""
        for i in range(self.decoder_layer_num):
            layer = self.decoder_layer_prefix + f"{i}"
            per_layer_out_ckpt = out_f.create_group(f"h_._{i}")

            for converted_layer_name, convert_fn in convert_dict.items():
                converted_np_array = convert_fn(state_dict, layer)
                per_layer_out_ckpt[converted_layer_name] = converted_np_array
                pbar.update()


class EncoderTFBlockConversionInterface(ABC):
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
        self,
        state_dict: Dict[str, torch.Tensor],
        convert_dict: Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]],
        out_f: h5py.Group,
        pbar: std_tqdm,
    ) -> None:
        """Return the number of transformer blocks converted in the encoder."""
        for i in range(self.encoder_layer_num):
            layer = self.encoder_layer_prefix + f"{i}"
            per_layer_out_ckpt = out_f.create_group(f"h_._{i}")

            for converted_layer_name, convert_fn in convert_dict.items():
                converted_np_array = convert_fn(state_dict, layer)
                per_layer_out_ckpt[converted_layer_name] = converted_np_array
                pbar.update()
