# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Checkpoint Converter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from peft import PeftType  # type: ignore[import] # pylint: disable=import-error
from peft.config import PeftConfig
from peft.tuners.lora import (  # type: ignore[import] # pylint: disable=import-error
    LoraConfig,
)
from transformers import GenerationConfig, PretrainedConfig  # type: ignore[import]

from friendli.enums import CheckpointDataType
from friendli.errors import NotSupportedCheckpointError
from friendli.logging import logger
from friendli.modules.converter.interface import (
    DecoderTFBlockConversionInterface,
    EncoderTFBlockConversionInterface,
    ModelConversionInterface,
    NonTFBlockConversionInterface,
)
from friendli.modules.converter.schema import ConvertInfo

SUPPORTED_GELU_FAMILY = [
    "gelu",
    "gelu_fast",
    "gelu_new",
    "gelu_python",
    "gelu_pytorch_tanh",
    "gelu_accurate",
]
SUPPORTED_HEAD_SIZE = [64, 80, 96, 128, 256]

MODEL_TYPE_TO_LORA_TARGET_MODULES_MAP = {
    "gptj": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
    "mpt": ["Wqkv"],
}

ENCODER_PREFIX = "encoder"
DECODER_PREFIX = "decoder"


class AbstractConverter(ModelConversionInterface, ABC):
    """Abstract class for converting Hugging Face checkpoint to Friendli checkpoint.

    Attributes:
        config (PreTrainedConfig): Hugging Face model configuration.
        generation_config (Optional[GenerationConfig]): Hugginface generation config.
            When set to None, `config` is used for configuring generation.
        data_type (CheckpointDataType): Data type for the Friendli checkpoint.
        quantize (bool): Whether to quantize the Friendli checkpoint.

    """

    def __init__(
        self,
        config: PretrainedConfig,
        generation_config: Optional[GenerationConfig],
        data_type: CheckpointDataType,
    ) -> None:
        """Initialize converter."""
        self.config = config
        self.generation_config = generation_config
        self.data_type = data_type

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

    def token_embed_weight_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape embedding layer's weight to Friendli format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The tensor of reshaped embedding weight.

        """
        assert len(params) == 1
        return params[0]

    def pos_embed_weight_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape position embedding layer's weight to Friendli format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The tensor of reshaped position embedding weight.
        """
        assert len(params) == 1
        return params[0]

    def head_weight_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape head layer's weight to Friendli format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The tensor of reshaped head weight.

        """
        assert len(params) == 1
        return params[0]

    def linear_weight_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape linear layer's weight to Friendli format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The tensor of reshaped linear weight.

        """
        assert len(params) == 1
        param = params[0].transpose(0, 1)
        return param

    def linear_bias_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape linear layer's bias to Friendli format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The tensor of reshaped linear bias.

        """
        assert len(params) == 1
        return params[0]

    def ln_weight_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape layer norm layer's weight to Friendli format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The tensor of reshaped layer norm weight.

        """
        assert len(params) == 1
        return params[0]

    def ln_bias_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape layer norm layer's bias to Friendli format.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The tensor of reshaped layer norm bias.

        """
        assert len(params) == 1
        return params[0]

    def qkv_weight_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape qkv layer's weight to Friendli format.

        In the original checkpoint, the qkv weight is stored as a single tensor or
        separated by three tensors. In the Friendli checkpoint, it is stored as a single tensor.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
            The tensor of reshaped qkv weight.

        """
        param = torch.cat(params, dim=0)
        param = param.transpose(0, 1)
        return param

    def qkv_bias_reshape(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Reshape qkv layer's bias to Friendli format.

        In the original checkpoint, the qkv weight is stored as a single tensor or
        separated by three tensors. In the Friendli checkpoint, it is stored as a single tensor.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state_dict of the original checkpoint.
            layer (str): The layer name of the original checkpoint.
            per_layer_postfixes (List[str]): The list of postfixes of the layer.

        Returns:
           The tensor of reshaped qkv bias.

        """
        param = torch.cat(params, dim=0)
        return param


class DecoderOnlyConverter(
    AbstractConverter,
    NonTFBlockConversionInterface,
    DecoderTFBlockConversionInterface,
):
    """Converter for Decoder-Only models."""

    def check_config(self) -> None:
        """Check if a convertible form of the checkpoint from the decoder-only model config."""
        super().check_config()
        if self.decoder_head_size not in SUPPORTED_HEAD_SIZE:
            raise NotSupportedCheckpointError(
                invalid_option=f"decoder_head_size={self.decoder_head_size}",
                valid_options=SUPPORTED_HEAD_SIZE,
            )

    def get_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Get List of conversion informations for Decoder-Only model."""
        return self.non_transformer_convert_info_list + self.decoder_convert_info_list


class EncoderDecoderConverter(
    AbstractConverter,
    NonTFBlockConversionInterface,
    EncoderTFBlockConversionInterface,
    DecoderTFBlockConversionInterface,
):
    """Converter for Encoder-Decoder models."""

    def check_config(self) -> None:
        """Check if a convertible form of the checkpoint from the encoder-decoder model config."""
        if self.decoder_head_size not in SUPPORTED_HEAD_SIZE:
            raise NotSupportedCheckpointError(
                invalid_option=f"decoder_head_size={self.decoder_head_size}",
                valid_options=SUPPORTED_HEAD_SIZE,
            )

    def get_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Get list of conversion informations for Encoder-Decoder model."""
        return (
            self.non_transformer_convert_info_list
            + self.decoder_convert_info_list
            + self.encoder_convert_info_list
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


class DecoderOnlyLoraConverter(AbstractConverter):
    """Converter for LoRA modules in the models."""

    def __init__(
        self,
        converter: AbstractConverter,
        adapter_config: PeftConfig,
    ) -> None:
        """Initialize LoRA Converter."""
        super().__init__(
            config=converter.config,
            generation_config=converter.generation_config,
            data_type=converter.data_type,
        )
        self.converter = cast(DecoderOnlyConverter, converter)
        self.adapter_config = cast(LoraConfig, adapter_config)

    def check_config(self) -> None:
        """Check if a convertible form of the checkpoint from the LoRAconfig."""
        if self.adapter_config.peft_type != PeftType.LORA:
            raise NotSupportedCheckpointError(
                invalid_option=f"peft_type={self.adapter_config.peft_type}",
                valid_options=[str(PeftType.LORA)],
            )
        if self.config.model_type not in MODEL_TYPE_TO_LORA_TARGET_MODULES_MAP:
            raise NotSupportedCheckpointError(
                invalid_option=f"model_type={self.config.model_type} for LORA",
                valid_options=list(MODEL_TYPE_TO_LORA_TARGET_MODULES_MAP.keys()),
            )
        if (
            self.adapter_config.layers_pattern is not None
            and len(self.adapter_config.layers_pattern) > 0
        ):
            raise NotSupportedCheckpointError(
                invalid_option=f"layers_pattern={self.adapter_config.layers_pattern}",
                valid_options=[None, [], ""],
            )
        if (
            self.adapter_config.rank_pattern is not None
            and len(self.adapter_config.rank_pattern) > 0
        ):
            raise NotSupportedCheckpointError(
                invalid_option=f"rank_pattern={self.adapter_config.rank_pattern}",
                valid_options=[None, {}],
            )
        if (
            self.adapter_config.alpha_pattern is not None
            and len(self.adapter_config.alpha_pattern) > 0
        ):
            raise NotSupportedCheckpointError(
                invalid_option=f"alpha_pattern={self.adapter_config.alpha_pattern}",
                valid_options=[None, {}],
            )
        if self.adapter_config.target_modules is None or set(
            self.adapter_config.target_modules
        ) != set(MODEL_TYPE_TO_LORA_TARGET_MODULES_MAP[self.config.model_type]):
            raise NotSupportedCheckpointError(
                invalid_option=f"target_modules={self.adapter_config.target_modules} for LORA",
                valid_options=[
                    str(MODEL_TYPE_TO_LORA_TARGET_MODULES_MAP[self.config.model_type])
                ],
            )
        if (self.adapter_config.layers_to_transform is not None) and (
            self.adapter_config != list(range(self.converter.decoder_layer_num))
        ):
            raise NotSupportedCheckpointError(
                invalid_option=f"layers_to_transform={self.adapter_config.layers_to_transform}",
                valid_options=[
                    f"layers_to_transform=None"
                    f"layers_to_transform={list(range(self.converter.decoder_layer_num))}",
                ],
            )

    def get_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Get convert dict for LoRA model."""
        return self.adapter_convert_info_list

    def _get_layers_to_transform(self) -> List[int]:
        layers_to_transform = cast(LoraConfig, self.adapter_config).layers_to_transform
        if layers_to_transform is None:
            layers_to_transform = list(range(self.converter.decoder_layer_num))
        else:
            if isinstance(layers_to_transform, int):
                layers_to_transform = [layers_to_transform]
        return layers_to_transform

    def lora_weight_reshape(
        self,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Reshape LoRA layer's weight to Friendli format."""
        assert len(params) == 1
        return params[0].transpose(0, 1)

    def pre_convert(self, model: torch.nn.Module) -> torch.nn.Module:
        """Preprocess the adapter modules before converting.

        All the parameters of the LoRA low-rank matrixs are converted by `lora_weight_reshape`.
        If the parameter can't be converted by `lora_weight_reshape`,

        """
        return model

    def convert(  # pylint: disable=too-many-locals
        self,
        model: torch.nn.Module,
        convert_info_list: List[ConvertInfo],
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Reshape Lora adapter model's all layer to Friendli format."""
        model = self.pre_convert(model)
        yield from self.converter.convert(model, convert_info_list)

    def get_attributes(self) -> Dict[str, Any]:
        """Get adapter checkpoint attributes."""
        return {
            "name": "FILL ME",
            "type": "lora",
            "alpha": self.adapter_config.lora_alpha,
            "rank": self.adapter_config.r,
            "target-modules": self.adapter_target_modules,
            "ckpt-path": "FILL ME",
        }

    @property
    @abstractmethod
    def adapter_target_modules(self) -> List[str]:
        """Return the target modules that LoRA applies to."""

    @property
    @abstractmethod
    def adapter_convert_info_list(
        self,
    ) -> List[ConvertInfo]:
        """Return the list of conversion informations for LoRA modules of the model."""


OneOfAdapterConverter = DecoderOnlyLoraConverter
OneOfConverter = Union[EncoderDecoderConverter, DecoderOnlyConverter]
