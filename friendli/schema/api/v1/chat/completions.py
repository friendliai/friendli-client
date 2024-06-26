# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

"""Friendli V1 Chat Completion Serving API Schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from typing_extensions import Required, TypedDict


class MessageParam(TypedDict, total=False):
    """Message param schema."""

    role: Required[str]
    content: Required[str]


class ToolFunctionParam(TypedDict, total=False):
    """Tool function param schema."""

    name: Required[str]
    description: Optional[str]
    parameters: Dict[str, Any]


class ToolParam(TypedDict, total=False):
    """Tool param schema."""

    type: Required[str]
    function: Required[ToolFunctionParam]


class ResponseFormatParam(TypedDict, total=True):
    """Response format param schema."""

    type: Required[str]
    schema: Optional[str]


class Function(BaseModel):
    """Function schema."""

    arguments: str
    name: str


class ChatCompletionMessageToolCall(BaseModel):
    """Tool call schema."""

    id: str
    function: Function
    type: Literal["function"]


class Message(BaseModel):
    """Message schema."""

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class TopLogprob(BaseModel):
    """Top logprob schema."""

    token: str
    bytes: Optional[List[int]] = None
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    """Chat completion token log prob schema."""

    token: str
    bytes: Optional[List[int]] = None
    logprob: float
    top_logprobs: List[TopLogprob]


class ChoiceLogprobs(BaseModel):
    """Schema of log prob info."""

    content: Optional[List[ChatCompletionTokenLogprob]] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice schema."""

    index: int
    message: Message
    finish_reason: str
    logprobs: Optional[ChoiceLogprobs] = None


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
