# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Schema of Endpoint Config."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pydantic


class GpuConfig(pydantic.BaseModel):
    """GPU config."""

    type: str
    count: int


class EndpointConfig(pydantic.BaseModel):
    """Endpoint config."""

    name: str
    gpu: GpuConfig
    model: str
    adapters: Optional[List[str]] = None
    launch_config: Optional[Dict[str, Any]] = None
