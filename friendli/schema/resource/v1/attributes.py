# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""V1 Checkpoint Attributes Schemas."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Extra, Field
from typing_extensions import Annotated

from friendli.enums import CheckpointDataType, QuantMode
from friendli.utils.compat import PYDANTIC_V2


class V1CommonAttributes(BaseModel):
    """V1 checkpoint attributes schema."""

    if PYDANTIC_V2:
        model_config = ConfigDict(protected_namespaces=(), extra=Extra.forbid)  # type: ignore
    else:
        model_config = ConfigDict(extra=Extra.forbid)

    dtype: CheckpointDataType
    quant_scheme: Optional[QuantMode] = None
    quant_group_size: Optional[int] = None
    quant_bit: Optional[int] = None


class V1BlenderbotAttributes(V1CommonAttributes):
    """V1 Blenderbot attributes schema."""

    model_type: Literal["blenderbot"]
    head_size: int
    num_heads: int
    hidden_size: int
    ff_intermediate_size: int
    num_encoder_layers: int
    num_decoder_layers: int
    max_input_length: int
    max_output_length: int
    vocab_size: int
    eos_token: int
    decoder_start_token: int


class V1BloomAttributes(V1CommonAttributes):
    """V1 Bloom attributes schema."""

    model_type: Literal["bloom"]
    head_size: int
    num_heads: int
    num_layers: int
    max_length: int
    vocab_size: int
    eos_token: int


class V1GPTAttributes(V1CommonAttributes):
    """V1 GPT attributes schema."""

    model_type: Literal["gpt"]
    head_size: int
    num_heads: int
    num_layers: int
    max_length: int
    vocab_size: int
    eos_token: int


class V1GPTJAttributes(V1CommonAttributes):
    """V1 GPT-J attributes schema."""

    model_type: Literal["gpt-j"]
    head_size: int
    rotary_dim: int
    num_heads: int
    num_layers: int
    max_length: int
    vocab_size: int
    eos_token: int
    rope_theta: Optional[float]


class V1GPTNeoXAttributes(V1CommonAttributes):
    """V1 GPT-NeoX attributes schema."""

    model_type: Literal["gpt-neox"]
    head_size: int
    rotary_dim: int
    num_heads: int
    num_layers: int
    max_length: int
    vocab_size: int
    eos_token: int
    rope_theta: Optional[float]


class V1GPTNeoXHFAttributes(V1CommonAttributes):
    """V1 GPT-NeoX HF attributes schema."""

    model_type: Literal["gpt-neox-hf"]
    head_size: int
    rotary_dim: int
    num_heads: int
    num_layers: int
    max_length: int
    vocab_size: int
    eos_token: int
    rope_theta: Optional[float]


class V1LlamaAttributes(V1CommonAttributes):
    """V1 LLaMA attributes schema."""

    model_type: Literal["llama"]
    head_size: int
    rotary_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    ff_intermediate_size: int
    max_length: int
    vocab_size: int
    eos_token: int
    rope_theta: Optional[float]


class V1MistralAttributes(V1CommonAttributes):
    """V1 Mistral attributes schema."""

    model_type: Literal["mistral"]
    head_size: int
    rotary_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    ff_intermediate_size: int
    max_length: int
    vocab_size: int
    eos_token: int
    attention_window_size: int


class V1OPTAttributes(V1CommonAttributes):
    """V1 OPT attributes schema."""

    model_type: Literal["opt"]
    head_size: int
    num_heads: int
    num_layers: int
    max_length: int
    vocab_size: int
    eos_token: int


class V1T5Attributes(V1CommonAttributes):
    """V1 T5 attributes schema."""

    model_type: Literal["t5", "t5-v1_1"]
    head_size: int
    num_heads: int
    hidden_size: int
    ff_intermediate_size: int
    num_encoder_layers: int
    num_decoder_layers: int
    max_input_length: int
    max_output_length: int
    num_pos_emb_buckets: int
    max_pos_distance: int
    vocab_size: int
    eos_token: int
    decoder_start_token: int


class V1FalconAttributes(V1CommonAttributes):
    """V1 Falcon attributes schema."""

    model_type: Literal["falcon-7b", "falcon"]
    head_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    max_length: int
    vocab_size: int
    eos_token: int
    rotary_dim: int
    rope_theta: Optional[float]


class V1MPTAttributes(V1CommonAttributes):
    """V1 MPT attributes schema."""

    model_type: Literal["mpt"]
    head_size: int
    num_heads: int
    num_layers: int
    max_length: int
    vocab_size: int
    eos_token: int
    clip_qkv: float
    num_kv_heads: Optional[int]


class V1PhiAttributes(V1CommonAttributes):
    """V1 Phi attributes schema."""

    model_type: Literal["phi"]
    head_size: int
    rotary_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    ff_intermediate_size: int
    max_length: int
    vocab_size: int
    eos_token: int
    rope_theta: Optional[float]


V1CheckpointAttributes = Annotated[
    Union[
        V1BlenderbotAttributes,
        V1BloomAttributes,
        V1FalconAttributes,
        V1GPTAttributes,
        V1GPTJAttributes,
        V1GPTNeoXAttributes,
        V1GPTNeoXHFAttributes,
        V1LlamaAttributes,
        V1MistralAttributes,
        V1MPTAttributes,
        V1OPTAttributes,
        V1T5Attributes,
        V1PhiAttributes,
    ],
    Field(discriminator="model_type"),
]


class V1AttributesValidationModel(BaseModel):
    """Model for validating attributes."""

    model_config = ConfigDict(extra=Extra.forbid)

    attr: V1CheckpointAttributes
