# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import ClassVar as _ClassVar
from typing import Optional as _Optional

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message

DESCRIPTOR: _descriptor.FileDescriptor

class V1TokenizeRequest(_message.Message):
    __slots__ = ["prompt"]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    def __init__(self, prompt: _Optional[str] = ...) -> None: ...
