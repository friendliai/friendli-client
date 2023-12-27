# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class V1ChatCompletionsRequest(_message.Message):
    __slots__ = [
        "messages",
        "frequency_penalty",
        "max_tokens",
        "n",
        "presence_penalty",
        "stop",
        "stream",
        "temperature",
        "top_p",
        "timeout_microseconds",
    ]

    class Message(_message.Message):
        __slots__ = ["content", "role"]
        CONTENT_FIELD_NUMBER: _ClassVar[int]
        ROLE_FIELD_NUMBER: _ClassVar[int]
        content: str
        role: str
        def __init__(
            self, content: _Optional[str] = ..., role: _Optional[str] = ...
        ) -> None: ...
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[
        V1ChatCompletionsRequest.Message
    ]
    frequency_penalty: float
    max_tokens: int
    n: int
    presence_penalty: float
    stop: _containers.RepeatedScalarFieldContainer[str]
    stream: bool
    temperature: float
    top_p: float
    timeout_microseconds: int
    def __init__(
        self,
        messages: _Optional[
            _Iterable[_Union[V1ChatCompletionsRequest.Message, _Mapping]]
        ] = ...,
        frequency_penalty: _Optional[float] = ...,
        max_tokens: _Optional[int] = ...,
        n: _Optional[int] = ...,
        presence_penalty: _Optional[float] = ...,
        stop: _Optional[_Iterable[str]] = ...,
        stream: bool = ...,
        temperature: _Optional[float] = ...,
        top_p: _Optional[float] = ...,
        timeout_microseconds: _Optional[int] = ...,
    ) -> None: ...
