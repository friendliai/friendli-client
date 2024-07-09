# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""GraphQL stub."""

from __future__ import annotations

from ..stub.protocol.httpx import SyncHttpxStub
from .api import BaseSyncStub


class GraphqlStub(SyncHttpxStub, BaseSyncStub):
    """Graphql stub."""
