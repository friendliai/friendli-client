# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""File digest backport."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

try:
    from hashlib import file_digest  # type: ignore[attr-defined]

    def file_sha256(file: PathLike) -> str:
        """Compute sha256 of a file."""
        with Path(file).open("rb") as f:
            digest = file_digest(f, "sha256").hexdigest()
            return f"sha256:{digest}"

except ImportError:

    chunk_size = 2**18

    def file_sha256(file: PathLike) -> str:
        """Compute sha256 of a file."""
        h = sha256()
        mv = memoryview(bytearray(chunk_size))

        with Path(file).open("rb") as f:
            while n := f.readinto(mv):
                h.update(mv[:n])
        digest = h.hexdigest()
        return f"sha256:{digest}"
