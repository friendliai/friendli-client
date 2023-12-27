# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Checkpoint Converter Interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from friendli.modules.converter.schema import ConvertInfo
from friendli.modules.converter.utils import (
    convert_tensor_to_np_array,
    get_tensor_from_state_dict,
)


class ModelConversionInterface(ABC):
    """Interface get information for converting models."""

    @abstractmethod
    def get_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Get list of conversion informations for the model."""

    @abstractmethod
    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""

    @abstractmethod
    def check_config(self) -> None:
        """Check if the model is convertable."""

    def convert(
        self,
        model: torch.nn.Module,
        convert_info_list: List[ConvertInfo],
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Convert Huggingface Model to Friendli format(.h5).

        Args:
            model (torch.nn.Module): Huggingface model.
            output_path (str): Path to save the converted checkpoint.
            convert_info_list (List[ConvertInfo]):
                List of convert information of the parameter in huggingface checkpoint.
        """
        state_dict = model.state_dict()
        total_layers = len(convert_info_list)
        with tqdm(total=total_layers, desc="Converting", unit="tensor") as pbar:
            for convert_info in convert_info_list:
                converted_name, reshape_fn, param_names, data_type = (
                    convert_info.converted_name,
                    convert_info.reshape_fn,
                    convert_info.param_names,
                    convert_info.data_type,
                )
                params = [
                    get_tensor_from_state_dict(state_dict, param_name)
                    for param_name in param_names
                ]
                reshaped_tensor = reshape_fn(params)
                yield (
                    converted_name,
                    convert_tensor_to_np_array(reshaped_tensor, data_type),
                )
                pbar.update()


class NonTFBlockConversionInterface(ABC):
    """Interface get information for converting common layers."""

    @property
    @abstractmethod
    def non_transformer_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for the non-transformer blocks."""


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
    def decoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for transformer blocks in the decoder."""

    @property
    @abstractmethod
    def decoder_head_size(self) -> int:
        """Return the head size of the decoder."""

    @property
    @abstractmethod
    def decoder_ff_intermediate_size(self) -> int:
        """Return the intermediate size of the linear layer in decoder's MLP."""


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
    def encoder_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for transformer blocks in the encoder."""

    @property
    @abstractmethod
    def encoder_head_size(self) -> int:
        """Return the head size of the encoder."""

    @property
    @abstractmethod
    def encoder_ff_intermediate_size(self) -> int:
        """Return the intermediate size of the linear layer in encoder's MLP."""


class RotaryEmbeddingConversionInterface(ABC):
    """Interface get information for converting rotary embeddings."""

    @property
    @abstractmethod
    def rotary_dim(self) -> int:
        """Return the dimension of rotary embeddings."""

    @property
    @abstractmethod
    def rotary_emb_base(self) -> float:
        """Return the base of rotary embeddings."""
