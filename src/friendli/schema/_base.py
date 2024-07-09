# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Schema base."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


def snake2camel(snake: str) -> str:
    """Convert a snake_case string to camelCase."""
    return "".join(
        _capitalize(word) if i > 0 else word for i, word in enumerate(snake.split("_"))
    )


def _capitalize(s: str) -> str:
    return s.upper() if len(s) == 2 and s[0].isdigit() else s.capitalize()


class ModelBase(BaseModel):
    """Base model for API models.

    Translates camelCase to snake_case.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        alias_generator=snake2camel,
        frozen=True,
    )
