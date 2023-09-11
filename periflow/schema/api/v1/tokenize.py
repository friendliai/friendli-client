# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow V1 Tokenize API Schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class V1TokenizeRequest(BaseModel):
    """V1 tokenize request schema."""

    prompt: Optional[str] = None


class V1TokenizeResponse(BaseModel):
    """V1 tokenize response schema."""

    tokens: List[int]
