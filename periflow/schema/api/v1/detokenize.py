# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow V1 Detokenize API Schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class V1DetokenizeRequest(BaseModel):
    """V1 detokenize request schema."""

    tokens: Optional[List[int]] = None


class V1DetokenizeResponse(BaseModel):
    """V1 detokenize response schema."""

    text: str
