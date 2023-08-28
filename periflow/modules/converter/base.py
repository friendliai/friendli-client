# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint Converter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import h5py  # type: ignore[import]
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import GenerationConfig, PretrainedConfig  # type: ignore[import]

from periflow.enums import CheckpointDataType
from periflow.errors import NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.interface import (
    ConversionInterface,
    DecoderConversionInterface,
    EncoderConversionInterface,
)
from periflow.modules.converter.utils import (
    convert_tensor_to_np_array,
    get_tensor_from_state_dict,
)

SUPPORTED_GELU_FAMILY = [
    "gelu",
    "gelu_fast",
    "gelu_new",
    "gelu_python",
    "gelu_pytorch_tanh",
    "gelu_accurate",
]
SUPPORTED_HEAD_SIZE = [64, 80, 96, 128, 256]


class AbstractConverter(ABC):
    """Abstract class for converting Hugging Face checkpoint to Periflow checkpoint.

    Attributes:
        config (PreTrainedConfig): Hugging Face model configuration.
        generation_config (Optional[GenerationConfig]): Hugginface generation config.
            When set to None, `config` is used for configuring generation.
        output_path (str): Output path for the Periflow checkpoint.
        data_type (CheckpointDataType): Data type for the Periflow checkpoint.

    """

    def __init__(
        self,
        config: PretrainedConfig,
        generation_config: Optional[GenerationConfig],
        output_path: str,
        data_type: CheckpointDataType,
    ) -> None:
        """Initialize converter."""
        self.config = config

        self.generation_config = generation_config
        self.output_path = output_path
        self.data_type = data_type

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Get model type."""

    @abstractmethod
    def check_config(self) -> None:
        """Check if the given model config can be converted to PeriFlow format."""

    @abstractmethod
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Convert all layers in state_dict to PeriFlow format."""

    @abstractmethod
    def get_attributes(self) -> Dict[str, Any]:
        """Get checkpoint attributes."""

    def get_eos_token_id(self) -> Optional[int]:
        """Get ID of EOS token."""
        generation_eos_token_id = None
        if self.generation_config is not None:
            generation_eos_token_id = self.generation_config.eos_token_id

        config_eos_token_id = self.config.eos_token_id

        if generation_eos_token_id is None:
            eos_token_id = config_eos_token_id
        else:
            if generation_eos_token_id != config_eos_token_id:
                logger.warn(
                    "'eos_token' is different in generation_config (%s) and config (%s). "
                    "Please fill the correct value.",
                    generation_eos_token_id,
                    config_eos_token_id,
                )
                eos_token_id = None
            else:
                eos_token_id = config_eos_token_id

        if eos_token_id is None:
            logger.warn(
                "'eos_token' cannot be automatically configured. "
                "Please fill in the field by yourself."
            )

        return eos_token_id

    def token_embed_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert embedding layer's weight to PeriFlow format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The numpy array of the converted embedding weight.

        """
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)

    def pos_embed_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert position embedding layer's weight to PeriFlow format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The numpy array of the converted position embedding weight.
        """
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)

    def head_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert head layer's weight to PeriFlow format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The numpy array of the converted head weight.

        """
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)

    def linear_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert linear layer's weight to PeriFlow format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The numpy array of the converted linear weight.

        """
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        param = param.transpose(0, 1)
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)

    def linear_bias_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert linear layer's bias to PeriFlow format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The numpy array of the converted linear bias.

        """
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)

    def ln_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert layer norm layer's weight to PeriFlow format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The numpy array of the converted layer norm weight.

        """
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)

    def ln_bias_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert layer norm layer's bias to PeriFlow format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The numpy array of the converted layer norm bias.

        """
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(state_dict, layer + per_layer_postfixes[0])
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)

    def qkv_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert qkv layer's weight to PeriFlow format.

        In the original checkpoint, the qkv weight is stored as a single tensor or
        separated by three tensors. In the PeriFlow checkpoint, it is stored as a single tensor.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The numpy array of the converted qkv weight.

        """
        params = [
            get_tensor_from_state_dict(state_dict, layer + postfix)
            for postfix in per_layer_postfixes
        ]
        param = torch.cat(params, dim=0)
        param = param.transpose(0, 1)
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)

    def qkv_bias_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert qkv layer's bias to PeriFlow format.

        In the original checkpoint, the qkv weight is stored as a single tensor or
        separated by three tensors. In the PeriFlow checkpoint, it is stored as a single tensor.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
           The numpy array of the converted qkv bias.

        """
        params = [
            get_tensor_from_state_dict(state_dict, layer + postfix)
            for postfix in per_layer_postfixes
        ]
        param = torch.cat(params, dim=0)
        return convert_tensor_to_np_array(param=param, data_type=self.data_type)


class DecoderOnlyConverter(
    AbstractConverter, ConversionInterface, DecoderConversionInterface
):
    """Converter for Decoder-Only models."""

    def check_config(self) -> None:
        """Check if a convertible form of the checkpoint from the decoder-only model config."""
        if self.decoder_head_size not in SUPPORTED_HEAD_SIZE:
            raise NotSupportedCheckpointError(
                invalid_option=f"decoder_head_size={self.decoder_head_size}",
                valid_options=SUPPORTED_HEAD_SIZE,
            )

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Convert Decoder-Only model's all layer to PeriFlow format."""
        total_layers = len(self.decoder_convert_dict) * self.decoder_layer_num + len(
            self.non_transformer_convert_dict
        )
        with h5py.File(self.output_path, "w") as out_f, tqdm(
            total=total_layers, desc="Converting", unit="tensor"
        ) as pbar:
            self.convert_decoder_layers(state_dict=state_dict, out_f=out_f, pbar=pbar)
            self.convert_non_transformer_layers(
                state_dict=state_dict, out_f=out_f, pbar=pbar
            )


class EncoderDecoderConverter(
    AbstractConverter,
    ConversionInterface,
    EncoderConversionInterface,
    DecoderConversionInterface,
):
    """Converter for Encoder-Decoder models."""

    def check_config(self) -> None:
        """Check if a convertible form of the checkpoint from the encoder-decoder model config."""
        if self.decoder_head_size not in SUPPORTED_HEAD_SIZE:
            raise NotSupportedCheckpointError(
                invalid_option=f"decoder_head_size={self.decoder_head_size}",
                valid_options=SUPPORTED_HEAD_SIZE,
            )

    def convert(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Convert Encoder-Decoder model's all layer to PeriFlow format."""
        total_layers = (
            len(self.encoder_convert_dict) * self.encoder_layer_num
            + len(self.decoder_convert_dict) * self.decoder_layer_num
            + len(self.non_transformer_convert_dict)
        )
        with h5py.File(self.output_path, "w") as out_f, tqdm(
            total=total_layers, desc="Converting", unit="tensor"
        ) as pbar:
            self.convert_encoder_layers(state_dict=state_dict, out_f=out_f, pbar=pbar)
            self.convert_decoder_layers(state_dict=state_dict, out_f=out_f, pbar=pbar)
            self.convert_non_transformer_layers(
                state_dict=state_dict, out_f=out_f, pbar=pbar
            )

    def get_decoder_start_token_id(self) -> Optional[int]:
        """Get ID of decoder start token."""
        generation_decoder_start_token_id = None
        if self.generation_config is not None:
            generation_decoder_start_token_id = (
                self.generation_config.decoder_start_token_id
            )

        config_decoder_start_token_id = self.config.decoder_start_token_id

        if generation_decoder_start_token_id is None:
            decoder_start_token_id = config_decoder_start_token_id
        else:
            if generation_decoder_start_token_id != config_decoder_start_token_id:
                logger.warn(
                    "'decoder_start_token_id' is different in generation_config "
                    "(%s) and config (%s). Please fill the correct value.",
                    generation_decoder_start_token_id,
                    config_decoder_start_token_id,
                )
                decoder_start_token_id = None
            else:
                decoder_start_token_id = config_decoder_start_token_id

        if decoder_start_token_id is None:
            logger.warn(
                "'decoder_start_token' cannot be automatically configured. "
                "Please fill in the field by yourself."
            )

        return decoder_start_token_id
