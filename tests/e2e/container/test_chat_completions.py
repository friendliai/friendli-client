# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Test chat completions API of Friendli Container."""

from __future__ import annotations

import json

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
    messages = []
    for msg in examples:
        messages.append(msg)
        print(f"USER: {msg['content']}")
        if enable_stream:
            print("ASSISTANT: ", end="")
            stream = client.chat.completions.create(
                messages=messages,
                stream=True,
                min_tokens=10,
                max_tokens=10,
                top_k=1,
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
                messages=messages,
                min_tokens=10,
                max_tokens=10,
                top_k=1,
            )
            content = chat.choices[0].message.content
            print(f"ASSISTANT: {content}")
            messages.append({"role": "assistant", "content": content})
            assert content, "Output content is empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_stream", [False, True])
async def test_chat_completions_async(async_client: AsyncFriendli, enable_stream: bool):
    messages = []
    for msg in examples:
        messages.append(msg)
        print(f"USER: {msg['content']}")
        if enable_stream:
            print("ASSISTANT: ", end="")
            stream = await async_client.chat.completions.create(
                messages=messages,
                stream=True,
                min_tokens=10,
                max_tokens=10,
                top_k=1,
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
                messages=messages,
                min_tokens=10,
                max_tokens=10,
                top_k=1,
            )
            content = chat.choices[0].message.content
            print(f"ASSISTANT: {content}")
            messages.append({"role": "assistant", "content": content})
            assert content, "Output content is empty"


def get_current_weather(location, unit="fahrenheit") -> str:
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


available_tools = {"get_current_weather": get_current_weather}

messages = [
    {
        "role": "user",
        "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
    }
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


@pytest.mark.parametrize("enable_stream", [False, True])
def test_tool_calling(client: Friendli, enable_stream: bool):
    if enable_stream:
        stream = client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
            stream=True,
            top_k=1,
        )

        tool_calls = []
        tool_call = {}
        for chunk in stream:
            if chunk.choices[0].delta.tool_calls is not None:
                tool_calls_delta = chunk.choices[0].delta.tool_calls[0]
                if tool_calls_delta.id is not None:
                    if tool_calls_delta.index > 0:
                        tool_calls.append(tool_call)
                    tool_call = {
                        "id": tool_calls_delta.id,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tool_calls_delta.function.name is not None:
                    tool_call["function"]["name"] = tool_calls_delta.function.name
                tool_call["function"][
                    "arguments"
                ] += tool_calls_delta.function.arguments

        if tool_call:
            tool_calls.append(tool_call)

        if len(tool_calls) == 0:
            pytest.skip("This model does not work well for tool calling.")

        messages.append({"role": "assistant", "tool_calls": tool_calls})

        for tool_call in tool_calls:
            fn_name = tool_call["function"]["name"]
            fn = available_tools[fn_name]
            kwargs = json.loads(tool_call["function"]["arguments"])
            res = fn(**kwargs)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": fn_name,
                    "content": res,
                }
            )

        stream = client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
            stream=True,
            top_k=1,
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
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
            top_k=1,
        )

        tool_calls = chat.choices[0].message.tool_calls

        if tool_calls is None:
            pytest.skip("This model does not work well for tool calling.")

        messages.append({"role": "assistant", "tool_calls": tool_calls})

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn = available_tools[fn_name]
            kwargs = json.loads(tool_call.function.arguments)
            print(kwargs)
            res = fn(**kwargs)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": res,
                }
            )

        chat = client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
            top_k=1,
        )
        content = chat.choices[0].message.content
        print(f"ASSISTANT: {content}")
        messages.append({"role": "assistant", "content": content})
        assert content, "Output content is empty"
