# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Client: CLI and SDK for the fastest generative AI serving."""

from __future__ import annotations

import os

from friendli.di.injector import set_default_modules
from friendli.di.modules import default_modules
from friendli.sdk.client import AsyncFriendli, Friendli, FriendliResource

api_key = os.environ.get("FRIENDLI_API_KEY")
org_id = os.environ.get("FRIENDLI_ORG_ID")
project_id = os.environ.get("FRIENDLI_PRJ_ID")

set_default_modules(default_modules)

__all__ = [
    "api_key",
    "org_id",
    "project_id",
    "AsyncFriendli",
    "Friendli",
    "FriendliResource",
]
