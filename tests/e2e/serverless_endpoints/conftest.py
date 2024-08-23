# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Common fixtures for Serverless Endpoints."""

# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

import os
from typing import AsyncIterator, Iterator

import pytest

from friendli import AsyncFriendli, Friendli


@pytest.fixture(scope="module", autouse=True)
def check_env_vars():
    required_vars = ["FRIENDLI_TOKEN", "MODEL_ID", "TEAM_ID"]
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        pytest.skip(
            f"Skipping tests because the following environment variables are missing: {', '.join(missing_vars)}"
        )


@pytest.fixture
def client() -> Iterator[Friendli]:
    with Friendli(
        token=os.environ["FRIENDLI_TOKEN"],
        team_id=os.environ["TEAM_ID"],
    ) as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncIterator[AsyncFriendli]:
    async with AsyncFriendli(
        token=os.environ["FRIENDLI_TOKEN"],
        team_id=os.environ["TEAM_ID"],
    ) as client:
        yield client
