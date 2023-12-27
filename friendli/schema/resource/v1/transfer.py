# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Transfer request and response schemas."""

from __future__ import annotations

from typing import List

from pydantic import AnyHttpUrl, BaseModel


class UploadTask(BaseModel):
    """Upload URL response schema."""

    path: str
    upload_url: AnyHttpUrl


class MultipartUploadUrlInfo(BaseModel):
    """Multipart upload URL info."""

    upload_url: AnyHttpUrl
    part_number: int


class MultipartUploadTask(BaseModel):
    """Multipart upload URL response schema."""

    path: str
    upload_id: str
    upload_urls: List[MultipartUploadUrlInfo]


class UploadedPartETag(BaseModel):
    """Schema of entity tag info of uploaded part."""

    etag: str
    part_number: int
