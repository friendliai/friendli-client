# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Friendli V1 Chat Completion Chunk Serving API Schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel
from typing_extensions import Literal

from friendli.schema.api.v1.chat.completions import ChatCompletionTokenLogprob


class ChoiceDeltaFunctionCall(BaseModel):
    arguments: Optional[str] = None
    name: Optional[str] = None


class ChoiceDeltaToolCallFunction(BaseModel):
    arguments: Optional[str] = None
    name: Optional[str] = None


class ChoiceDeltaToolCall(BaseModel):
    index: int
    id: Optional[str] = None
    function: Optional[ChoiceDeltaToolCallFunction] = None
    type: Optional[Literal["function"]] = None


class ChoiceDelta(BaseModel):
    content: Optional[str] = None
    function_call: Optional[ChoiceDeltaFunctionCall] = None
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None


class ChoiceLogprobs(BaseModel):
    content: Optional[List[ChatCompletionTokenLogprob]] = None


class Choice(BaseModel):
    delta: ChoiceDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None
    index: int
    logprobs: Optional[ChoiceLogprobs] = None


class ChatCompletionChunk(BaseModel):
    choices: List[Choice]
    created: int
