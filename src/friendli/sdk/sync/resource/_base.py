# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Sync resource base."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import SyncClientBase


class ResourceBase:
    """Resource base."""

    def __init__(self, sdk: SyncClientBase) -> None:
        """Initialize auth resource."""
        self._sdk = sdk
