# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Common fixtures for Friendli Container."""

# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

import os
from typing import AsyncIterator, Iterator

import pytest

from friendli import AsyncFriendli, Friendli


def check_http_env_vars():
    required_vars = ["CONTAINER_HTTP_BASE_URL"]
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        pytest.skip(
            f"Skipping tests because the following environment variables are missing: {', '.join(missing_vars)}"
        )


def check_grpc_env_vars():
    required_vars = ["CONTAINER_GRPC_BASE_URL"]
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        pytest.skip(
            f"Skipping tests because the following environment variables are missing: {', '.join(missing_vars)}"
        )


@pytest.fixture
def client() -> Iterator[Friendli]:
    check_http_env_vars()
    with Friendli(base_url=os.environ["CONTAINER_HTTP_BASE_URL"]) as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncIterator[AsyncFriendli]:
    check_http_env_vars()
    async with AsyncFriendli(base_url=os.environ["CONTAINER_HTTP_BASE_URL"]) as client:
        yield client


@pytest.fixture
def grpc_client() -> Iterator[Friendli]:
    check_grpc_env_vars()
    with Friendli(
        base_url=os.environ["CONTAINER_GRPC_BASE_URL"], use_grpc=True
    ) as client:
        yield client


@pytest.fixture
async def async_grpc_client() -> AsyncIterator[AsyncFriendli]:
    check_grpc_env_vars()
    async with AsyncFriendli(
        base_url=os.environ["CONTAINER_GRPC_BASE_URL"], use_grpc=True
    ) as client:
        yield client
