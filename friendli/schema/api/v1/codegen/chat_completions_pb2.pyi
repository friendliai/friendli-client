# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers

from friendli.schema.api.v1.codegen import response_format_pb2 as _response_format_pb2

DESCRIPTOR: _descriptor.FileDescriptor

class ToolCall(_message.Message):
    __slots__ = ("id", "type", "function")

    class Function(_message.Message):
        __slots__ = ("name", "arguments")
        NAME_FIELD_NUMBER: _ClassVar[int]
        ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
        name: str
        arguments: str
        def __init__(
            self, name: _Optional[str] = ..., arguments: _Optional[str] = ...
        ) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: str
    function: ToolCall.Function
    def __init__(
        self,
        id: _Optional[str] = ...,
        type: _Optional[str] = ...,
        function: _Optional[_Union[ToolCall.Function, _Mapping]] = ...,
    ) -> None: ...

class Message(_message.Message):
    __slots__ = ("content", "role", "name", "tool_call_id", "tool_calls")
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    content: str
    role: str
    name: str
    tool_call_id: str
    tool_calls: _containers.RepeatedCompositeFieldContainer[ToolCall]
    def __init__(
        self,
        content: _Optional[str] = ...,
        role: _Optional[str] = ...,
        name: _Optional[str] = ...,
        tool_call_id: _Optional[str] = ...,
        tool_calls: _Optional[_Iterable[_Union[ToolCall, _Mapping]]] = ...,
    ) -> None: ...

class Tool(_message.Message):
    __slots__ = ("type", "function")

    class Function(_message.Message):
        __slots__ = ("name", "description", "parameters")
        NAME_FIELD_NUMBER: _ClassVar[int]
        DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
        PARAMETERS_FIELD_NUMBER: _ClassVar[int]
        name: str
        description: str
        parameters: _struct_pb2.Struct
        def __init__(
            self,
            name: _Optional[str] = ...,
            description: _Optional[str] = ...,
            parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        ) -> None: ...
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    type: str
    function: Tool.Function
    def __init__(
        self,
        type: _Optional[str] = ...,
        function: _Optional[_Union[Tool.Function, _Mapping]] = ...,
    ) -> None: ...

class V1ChatCompletionsRequest(_message.Message):
    __slots__ = (
        "messages",
        "model",
        "frequency_penalty",
        "logit_bias",
        "min_tokens",
        "max_tokens",
        "n",
        "presence_penalty",
        "stop",
        "stream",
        "temperature",
        "top_p",
        "timeout_microseconds",
        "logprobs",
        "top_logprobs",
        "top_k",
        "repetition_penalty",
        "seed",
        "eos_token",
        "tools",
        "response_format",
        "tool_choice",
        "parallel_tool_calls",
    )

    class LogitBiasEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: float
        def __init__(
            self, key: _Optional[int] = ..., value: _Optional[float] = ...
        ) -> None: ...
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    LOGIT_BIAS_FIELD_NUMBER: _ClassVar[int]
    MIN_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
    LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    TOP_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOOLS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    TOOL_CHOICE_FIELD_NUMBER: _ClassVar[int]
    PARALLEL_TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[Message]
    model: str
    frequency_penalty: float
    logit_bias: _containers.ScalarMap[int, float]
    min_tokens: int
    max_tokens: int
    n: int
    presence_penalty: float
    stop: _containers.RepeatedScalarFieldContainer[str]
    stream: bool
    temperature: float
    top_p: float
    timeout_microseconds: int
    logprobs: bool
    top_logprobs: int
    top_k: int
    repetition_penalty: float
    seed: _containers.RepeatedScalarFieldContainer[int]
    eos_token: _containers.RepeatedScalarFieldContainer[int]
    tools: _containers.RepeatedCompositeFieldContainer[Tool]
    response_format: _response_format_pb2.ResponseFormat
    tool_choice: _struct_pb2.Value
    parallel_tool_calls: bool
    def __init__(
        self,
        messages: _Optional[_Iterable[_Union[Message, _Mapping]]] = ...,
        model: _Optional[str] = ...,
        frequency_penalty: _Optional[float] = ...,
        logit_bias: _Optional[_Mapping[int, float]] = ...,
        min_tokens: _Optional[int] = ...,
        max_tokens: _Optional[int] = ...,
        n: _Optional[int] = ...,
        presence_penalty: _Optional[float] = ...,
        stop: _Optional[_Iterable[str]] = ...,
        stream: bool = ...,
        temperature: _Optional[float] = ...,
        top_p: _Optional[float] = ...,
        timeout_microseconds: _Optional[int] = ...,
        logprobs: bool = ...,
        top_logprobs: _Optional[int] = ...,
        top_k: _Optional[int] = ...,
        repetition_penalty: _Optional[float] = ...,
        seed: _Optional[_Iterable[int]] = ...,
        eos_token: _Optional[_Iterable[int]] = ...,
        tools: _Optional[_Iterable[_Union[Tool, _Mapping]]] = ...,
        response_format: _Optional[
            _Union[_response_format_pb2.ResponseFormat, _Mapping]
        ] = ...,
        tool_choice: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...,
        parallel_tool_calls: bool = ...,
    ) -> None: ...
