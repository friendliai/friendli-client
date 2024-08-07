# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Special types."""

from __future__ import annotations

from typing import NewType, Union

from pydantic import BaseModel, FilePath

TypeName = NewType("TypeName", str)


class UploadFile(BaseModel):
    """File abstraction for stub."""

    file: Union[bytes, FilePath]
