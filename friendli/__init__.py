# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Client: CLI and SDK for the fastest generative AI serving."""

from __future__ import annotations

import os

from friendli.di.injector import set_default_modules
from friendli.di.modules import default_modules
from friendli.sdk.client import AsyncFriendli, Friendli

token = os.environ.get("FRIENDLI_TOKEN")
team_id = os.environ.get("FRIENDLI_TEAM")
project_id = os.environ.get("FRIENDLI_PROJECT")

set_default_modules(default_modules)

__all__ = [
    "token",
    "team_id",
    "project_id",
    "AsyncFriendli",
    "Friendli",
]
