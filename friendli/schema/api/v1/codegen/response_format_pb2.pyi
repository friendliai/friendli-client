# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import ClassVar as _ClassVar
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class ResponseFormat(_message.Message):
    __slots__ = ("type", "schema")

    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        text: _ClassVar[ResponseFormat.Type]
        json_object: _ClassVar[ResponseFormat.Type]
        regex: _ClassVar[ResponseFormat.Type]
    text: ResponseFormat.Type
    json_object: ResponseFormat.Type
    regex: ResponseFormat.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    type: ResponseFormat.Type
    schema: str
    def __init__(
        self,
        type: _Optional[_Union[ResponseFormat.Type, str]] = ...,
        schema: _Optional[str] = ...,
    ) -> None: ...
