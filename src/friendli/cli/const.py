# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Constants."""

from __future__ import annotations

from enum import Enum

APP_NAME = "friendli-suite"
SERVICE_URL = "https://suite.friendli.ai"
SUITE_PAT_URL = "https://suite.friendli.ai/user-settings/tokens"


# Command group names
class Panel(str, Enum):
    """Panel names."""

    COMMON = "Common Commands"
    INFERENCE = "Inference Commands"
    DEDICATED = "Dedicated Endpoints Commands"
    SERVERLESS = "Serverless Endpoints Commands"
    CONTAINER = "Container Endpoints Commands"
    OTHER = "Other"
