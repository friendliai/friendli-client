# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Checkpoint Converter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union, cast

import h5py  # type: ignore[import]
import numpy as np
import torch
from peft import PeftType  # type: ignore[import] # pylint: disable=import-error
from peft.config import PeftConfig
from peft.tuners.lora import (  # type: ignore[import] # pylint: disable=import-error
    LoraConfig,
)
from tqdm.auto import tqdm
from transformers import GenerationConfig, PretrainedConfig  # type: ignore[import]

from periflow.enums import CheckpointDataType
from periflow.errors import NotSupportedCheckpointError
from periflow.logging import logger
from periflow.modules.converter.interface import (
    DecoderTFBlockConversionInterface,
    EncoderTFBlockConversionInterface,
    ModelConversionInterface,
    NonTFBlockConversionInterface,
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

MODEL_TYPE_TO_LORA_TARGET_MODULES_MAP = {
    "gptj": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
}

ENCODER_PREFIX = "encoder"
DECODER_PREFIX = "decoder"


class AbstractConverter(ModelConversionInterface, ABC):
    """Abstract class for converting Hugging Face checkpoint to Periflow checkpoint.

    Attributes:
        config (PreTrainedConfig): Hugging Face model configuration.
        generation_config (Optional[GenerationConfig]): Hugginface generation config.
            When set to None, `config` is used for configuring generation.
        data_type (CheckpointDataType): Data type for the Periflow checkpoint.
        quantize (bool): Whether to quantize the Periflow checkpoint.

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

    def get_convert_dict(
        self,
    ) -> Dict[str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]]:
        """Get convert dict for Decoder-Only model."""
        return {
            "non-transformer": self.non_transformer_convert_dict,
            "decoder": self.decoder_convert_dict,
        }

    def convert(
        self,
        model: torch.nn.Module,
        output_path: str,
        convert_dict: Dict[
            str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]
        ],
    ) -> None:
        """Convert Decoder-Only model's all layer to PeriFlow format."""
        state_dict = model.state_dict()
        total_layers = len(convert_dict["decoder"]) * self.decoder_layer_num + len(
            convert_dict["non-transformer"]
        )
        with h5py.File(output_path, "w") as out_f, tqdm(
            total=total_layers, desc="Converting", unit="tensor"
        ) as pbar:
            self.convert_decoder_layers(
                state_dict=state_dict,
                convert_dict=convert_dict["decoder"],
                out_f=out_f.create_group(DECODER_PREFIX),
                pbar=pbar,
            )
            self.convert_non_transformer_layers(
                state_dict=state_dict,
                convert_dict=convert_dict["non-transformer"],
                out_f=out_f,
                pbar=pbar,
            )


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

    def get_convert_dict(
        self,
    ) -> Dict[str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]]:
        """Get convert dict for Encoder-Decoder model."""
        return {
            "non-transformer": self.non_transformer_convert_dict,
            "encoder": self.encoder_convert_dict,
            "decoder": self.decoder_convert_dict,
        }

    def convert(
        self,
        model: torch.nn.Module,
        output_path: str,
        convert_dict: Dict[
            str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]
        ],
    ) -> None:
        """Convert Encoder-Decoder model's all layer to PeriFlow format."""
        state_dict = model.state_dict()
        total_layers = (
            len(self.encoder_convert_dict) * self.encoder_layer_num
            + len(self.decoder_convert_dict) * self.decoder_layer_num
            + len(self.non_transformer_convert_dict)
        )
        with h5py.File(output_path, "w") as out_f, tqdm(
            total=total_layers, desc="Converting", unit="tensor"
        ) as pbar:
            self.convert_decoder_layers(
                state_dict=state_dict,
                convert_dict=convert_dict["decoder"],
                out_f=out_f.create_group(DECODER_PREFIX),
                pbar=pbar,
            )
            self.convert_encoder_layers(
                state_dict=state_dict,
                convert_dict=convert_dict["encoder"],
                out_f=out_f.create_group(ENCODER_PREFIX),
                pbar=pbar,
            )
            self.convert_non_transformer_layers(
                state_dict=state_dict,
                convert_dict=convert_dict["non-transformer"],
                out_f=out_f,
                pbar=pbar,
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

    def get_convert_dict(
        self,
    ) -> Dict[str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]]:
        """Get convert dict for LoRA model."""
        return {
            "decoder": self.adapter_convert_dict,
        }

    def _get_layers_to_transform(self) -> List[int]:
        layers_to_transform = cast(LoraConfig, self.adapter_config).layers_to_transform
        if layers_to_transform is None:
            layers_to_transform = list(range(self.converter.decoder_layer_num))
        else:
            if isinstance(layers_to_transform, int):
                layers_to_transform = [layers_to_transform]
        return layers_to_transform

    def lora_weight_convert(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer_prefix: str,
        per_layer_postfixes: List[str],
    ) -> np.ndarray:
        """Convert LoRA layer's weight to PeriFlow format."""
        assert len(per_layer_postfixes) == 1
        param = get_tensor_from_state_dict(
            state_dict, layer_prefix + per_layer_postfixes[0]
        ).transpose(0, 1)
        return convert_tensor_to_np_array(
            param=param, data_type=self.converter.data_type
        )

    def pre_convert(self, model: torch.nn.Module) -> torch.nn.Module:
        """Preprocess the adapter modules before converting.

        All the parameters of the LoRA low-rank matrixs are converted by `lora_weight_convert`.
        If the parameter can't be converted by `lora_weight_convert`,

        """
        return model

    def convert(  # pylint: disable=too-many-locals
        self,
        model: torch.nn.Module,
        output_path: str,
        convert_dict: Dict[
            str, Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]
        ],
    ) -> None:
        """Convert Lora adapter model's all layer to PeriFlow format."""
        model = self.pre_convert(model)
        state_dict = model.state_dict()
        layers_to_transform = self._get_layers_to_transform()
        total_params = len(convert_dict["decoder"]) * len(layers_to_transform)

        with h5py.File(output_path, "w") as out_f, tqdm(
            total=total_params, desc="Converting", unit="tensor"
        ) as pbar:
            # TODO : need to support encoder-decoder modules.
            out_group = out_f.create_group(DECODER_PREFIX)
            for idx in layers_to_transform:
                layer_prefix = f"{self.converter.decoder_layer_prefix}{idx}"
                per_layer_out_ckpt = out_group.create_group(f"h_._{idx}")
                per_lora_out_ckpt = per_layer_out_ckpt.create_group("lora")
                for converted_param_name, convert_fn in convert_dict["decoder"].items():
                    converted_params = convert_fn(state_dict, layer_prefix)
                    per_lora_out_ckpt[converted_param_name] = converted_params
                    pbar.update()

    def get_attributes(self) -> Dict[str, Any]:
        """Get adapter checkpoint attributes."""
        return {
            "name": "FILL ME",
            "type": "lora",
            "alpha": self.adapter_config.lora_alpha,
            "rank": self.adapter_config.r,
            "ckpt-path": "FILL ME",
        }

    @property
    @abstractmethod
    def adapter_convert_dict(
        self,
    ) -> Dict[str, Callable[[Dict[str, torch.Tensor], str], np.ndarray]]:
        """Return the convert_dict for LoRA modules of the model."""


OneOfAdapterConverter = DecoderOnlyLoraConverter
OneOfConverter = Union[EncoderDecoderConverter, DecoderOnlyConverter]
