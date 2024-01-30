# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Image Schemas."""

from __future__ import annotations

from typing import List, Literal, Union

from pydantic import AnyHttpUrl, Base64Bytes, BaseModel, Field
from typing_extensions import TypeAlias

ImageResponseFormatParam: TypeAlias = Union[str, Literal["url", "png", "jpeg", "raw"]]


class ImageDataUrl(BaseModel):
    """Image data in url format."""

    format: Literal["url"]
    seed: int
    url: AnyHttpUrl


class ImageDataB64(BaseModel):
    """Image data in png format."""

    format: Literal["png", "jpeg", "raw"]
    seed: int
    b64_json: Base64Bytes


class Image(BaseModel):
    """Image data."""

    data: List[Union[ImageDataUrl, ImageDataB64]] = Field(..., discriminator="format")
