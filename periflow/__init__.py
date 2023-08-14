# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Client: CLI and SDK for the fastest generative AI serving."""

from __future__ import annotations

import os

from periflow.di.injector import set_default_modules
from periflow.di.modules import default_modules
from periflow.schema.api.v1.completion import V1CompletionOptions
from periflow.sdk.api.completion import Completion
from periflow.sdk.init import init
from periflow.sdk.resource.checkpoint import Checkpoint
from periflow.sdk.resource.credential import Credential
from periflow.sdk.resource.deployment import Deployment

api_key = os.environ.get("PERIFLOW_API_KEY")
org_id = os.environ.get("PERIFLOW_ORG_ID")
project_id = os.environ.get("PERIFLOW_PRJ_ID")

set_default_modules(default_modules)

__all__ = [
    "api_key",
    "org_id",
    "project_id",
    "init",
    "Checkpoint",
    "Credential",
    "Deployment",
    "Completion",
    "V1CompletionOptions",
]
