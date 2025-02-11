# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli V1 Completion Serving API Schemas."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel
from typing_extensions import Required, TypeAlias, TypedDict

BeamSearchType: TypeAlias = Literal["DETERMINISTIC", "STOCHASTIC", "NAIVE_SAMPLING"]


class TokenSequenceParam(TypedDict, total=False):
    """Token sequence param schema."""

    tokens: Required[List[int]]


class CompletionChoice(BaseModel):
    """Completion choice schema."""

    index: int
    seed: int
    text: str
    tokens: List[int]


class CompletionUsage(BaseModel):
    """Completion usage schema."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Completion(BaseModel):
    """Completion schema."""

    choices: List[CompletionChoice]
    usage: CompletionUsage


class CompletionLine(BaseModel):
    """Completion line schema."""

    event: str
    index: int = 0
    text: str = ""
    token: Optional[int]
    soft_prompt_ids: List[int] = []


class SoftPrompt(BaseModel):
    """Soft prompt param schema."""

    token_index_start: int
    token_index_end: int
    embeddings: List[float]
    id: Optional[int]
