# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Friendli V1 Chat Completion Serving API Schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel
from typing_extensions import Required, TypedDict


class MessageParam(TypedDict, total=False):
    """Message param schema."""

    role: Required[str]
    content: Required[str]


class Message(BaseModel):
    """Message schema."""

    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice schema."""

    index: int
    message: Message
    finish_reason: str


class ChatCompletionDeltaChoice(BaseModel):
    """Schema of chat completion choice with delta."""

    index: int
    delta: Message
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Chat completion usage schema."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """Chat completion schema."""

    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    created: int


class ChatCompletionLine(BaseModel):
    """Chat completion line schema."""

    choices: List[ChatCompletionDeltaChoice]
    created: int
