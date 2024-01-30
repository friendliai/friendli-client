# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import ClassVar as _ClassVar
from typing import Optional as _Optional

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message

DESCRIPTOR: _descriptor.FileDescriptor

class V1TextToImageRequest(_message.Message):
    __slots__ = [
        "prompt",
        "negative_prompt",
        "num_outputs",
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "response_format",
    ]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    NEGATIVE_PROMPT_FIELD_NUMBER: _ClassVar[int]
    NUM_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    NUM_INFERENCE_STEPS_FIELD_NUMBER: _ClassVar[int]
    GUIDANCE_SCALE_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    negative_prompt: str
    num_outputs: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    response_format: str
    def __init__(
        self,
        prompt: _Optional[str] = ...,
        negative_prompt: _Optional[str] = ...,
        num_outputs: _Optional[int] = ...,
        num_inference_steps: _Optional[int] = ...,
        guidance_scale: _Optional[float] = ...,
        seed: _Optional[int] = ...,
        response_format: _Optional[str] = ...,
    ) -> None: ...
