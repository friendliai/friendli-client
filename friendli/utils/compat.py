# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Compatibility Utils."""

from __future__ import annotations

from typing import Any, Dict, Type, TypeVar, cast

import pydantic

_ModelT = TypeVar("_ModelT", bound=pydantic.BaseModel)

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")


def model_parse(model: Type[_ModelT], data: Any) -> _ModelT:
    """Parse a pydantic model from data."""
    if PYDANTIC_V2:
        return model.model_validate(data)  # type: ignore
    return model.parse_obj(data)  # type: ignore


def model_dump(
    model: pydantic.BaseModel,
    *,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
) -> dict[str, Any]:
    """Dump data from a pydantic model."""
    if PYDANTIC_V2:
        return model.model_dump(  # type: ignore
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
        )
    return cast(
        Dict[str, Any],
        model.dict(  # type: ignore
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
        ),
    )
