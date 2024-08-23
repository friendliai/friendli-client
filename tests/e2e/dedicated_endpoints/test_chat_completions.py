# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Test chat completions API of Friendli Dedicated Endpoints."""

from __future__ import annotations

import os

import pytest

from friendli import Friendli
from friendli.sdk.client import AsyncFriendli

examples = [
    {
        "content": "These instructions apply to section-based themes (Responsive 6.0+, Retina 4.0+, Parallax 3.0+ Turbo 2.0+, Mobilia 5.0+). What theme version am I using?\nOn your Collections pages & Featured Collections sections, you can easily show the secondary image of a product on hover by enabling one of the theme's built-in settings!\nYour Collection pages & Featured Collections sections will now display the secondary product image just by hovering over that product image thumbnail.\nDoes this feature apply to all sections of the theme or just specific ones as listed in the text material?",
        "role": "user",
    },
    {
        "content": "Can you guide me through the process of enabling the secondary image hover feature on my Collection pages and Featured Collections sections?",
        "role": "user",
    },
    {
        "content": "Can you provide me with a link to the documentation for my theme?",
        "role": "user",
    },
    {
        "content": "Can you confirm if this feature also works for the Quick Shop section of my theme?",
        "role": "user",
    },
]


@pytest.mark.parametrize("enable_stream", [False, True])
def test_chat_completions(client: Friendli, enable_stream: bool):
    model = os.environ["ENDPOINT_ID"]
    messages = []
    for msg in examples:
        messages.append(msg)
        print(f"USER: {msg['content']}")
        if enable_stream:
            print("ASSISTANT: ", end="")
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                min_tokens=10,
                max_tokens=10,
            )
            content = ""
            for chunk in stream:
                chunk_content = chunk.choices[0].delta.content or ""
                content += chunk_content
                print(chunk_content, end="", flush=True)
            messages.append({"role": "assistant", "content": content})
            print()  # Add newline after stream output
            assert content, "Output content is empty"
        else:
            chat = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                min_tokens=10,
                max_tokens=10,
            )
            content = chat.choices[0].message.content
            print(f"ASSISTANT: {content}")
            messages.append({"role": "assistant", "content": content})
            assert content, "Output content is empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_stream", [False, True])
async def test_chat_completions_async(async_client: AsyncFriendli, enable_stream: bool):
    model = os.environ["ENDPOINT_ID"]
    messages = []
    for msg in examples:
        messages.append(msg)
        print(f"USER: {msg['content']}")
        if enable_stream:
            print("ASSISTANT: ", end="")
            stream = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                min_tokens=10,
                max_tokens=10,
            )
            content = ""
            async for chunk in stream:
                chunk_content = chunk.choices[0].delta.content or ""
                content += chunk_content
                print(chunk_content, end="", flush=True)
            messages.append({"role": "assistant", "content": content})
            print()  # Add newline after stream output
            assert content, "Output content is empty"
        else:
            chat = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                min_tokens=10,
                max_tokens=10,
            )
            content = chat.choices[0].message.content
            print(f"ASSISTANT: {content}")
            messages.append({"role": "assistant", "content": content})
            assert content, "Output content is empty"
