# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Test completions API of Friendli Serverless Endpoints."""

from __future__ import annotations

import os

import pytest

from friendli import Friendli
from friendli.sdk.client import AsyncFriendli

PROMPT = "Write me a function in Python that returns the fibonacci sequence."


@pytest.mark.parametrize("enable_stream", [False, True])
def test_chat_completions(client: Friendli, enable_stream: bool):
    model = os.environ["MODEL_ID"]
    print(f"PROMPT: {PROMPT}")
    if enable_stream:
        print("OUTPUT: ", end="")
        stream = client.completions.create(
            prompt=PROMPT,
            model=model,
            stream=True,
            min_tokens=10,
            max_tokens=10,
        )
        content = ""
        for chunk in stream:
            chunk_content = chunk.text or ""
            content += chunk_content
            print(chunk_content, end="", flush=True)
        print()  # Add newline after stream output
        assert content, "Output content is empty"
    else:
        chat = client.completions.create(
            prompt=PROMPT,
            model=model,
            min_tokens=10,
            max_tokens=10,
        )
        content = chat.choices[0].text
        print(f"OUTPUT: {content}")
        assert content, "Output content is empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_stream", [False, True])
async def test_chat_completions_async(async_client: AsyncFriendli, enable_stream: bool):
    model = os.environ["MODEL_ID"]
    print(f"PROMPT: {PROMPT}")
    if enable_stream:
        print("OUTPUT: ", end="")
        stream = await async_client.completions.create(
            prompt=PROMPT,
            model=model,
            stream=True,
            min_tokens=10,
            max_tokens=10,
        )
        content = ""
        async for chunk in stream:
            chunk_content = chunk.text or ""
            content += chunk_content
            print(chunk_content, end="", flush=True)
        print()  # Add newline after stream output
        assert content, "Output content is empty"
    else:
        chat = await async_client.completions.create(
            prompt=PROMPT,
            model=model,
            min_tokens=10,
            max_tokens=10,
        )
        content = chat.choices[0].text
        print(f"OUTPUT: {content}")
        assert content, "Output content is empty"
