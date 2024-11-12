<!---
Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.
-->

<p align="center">
  <img width="10%" alt="Friendli Logo" src="https://friendli.ai/icon.svg">
</p>

<h2><p align="center">Supercharge Generative AI Serving with Friendli 🚀</p></h2>

<p align="center">
  <a href="https://github.com/friendliai/friendli-client/actions/workflows/ci.yaml">
    <img alt="CI Status" src="https://github.com/friendliai/friendli-client/actions/workflows/ci.yaml/badge.svg">
  </a>
  <a href="https://pypi.org/project/friendli-client/">
    <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/friendli-client?logo=Python&logoColor=white">
  </a>
  <a href="https://pypi.org/project/friendli-client/">
      <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/friendli-client?logo=PyPI&logoColor=white">
  </a>
  <a href="https://friendli.ai/docs/">
    <img alt="Documentation" src="https://img.shields.io/badge/read-doc-blue?logo=ReadMe&logoColor=white">
  </a>
  <a href="https://github.com/friendliai/friendli-client/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=Apache">
  </a>
</p>

The Friendli Client offers convenient interface to interact with endpoint services provided by [Friendli Suite](https://suite.friendli.ai/), the ultimate solution for serving generative AI models. Designed for flexibility and performance, it supports both synchronous and asynchronous operations, making it easy to integrate powerful AI capabilities into your applications.

# Installation

To get started with Friendli, install the client package using `pip`:

```sh
pip install friendli-client
```

> [!IMPORTANT]
> You must set `FRIENDLI_TOKEN` environment variable before initializing the client instance with `client = Friendli()`.
> Alternatively, you can provide the value of your personal access token as the `token` argument when creating the client, like so:
>
> ```python
> from friendli import Friendli
>
> client = Friendli(token="YOUR PERSONAL ACCESS TOKEN")
> ```

# Friendli Serverless Endpoints

[Friendli Serverless Endpoint](https://friendli.ai/products/serverless-endpoints) offer a simple, click-and-play interface for accessing popular open-source models like Llama 3.1.
With pay-per-token billing, this is ideal for exploration and experimentation.

To interact with models hosted by serverless endpoints, provide the model code you want to use in the `model` argument. Refer to the [pricing table](https://friendli.ai/docs/guides/serverless_endpoints/pricing/) for a list of available model codes and their pricing.

```python
from friendli import Friendli

client = Friendli()

chat_completion = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
)
print(chat_completion.choices[0].message.content)
```

# Friendli Dedicated Endpoints

[Friendli Dedicated Endpoints](https://friendli.ai/products/dedicated-endpoints) enable you to run your custom generative AI models on dedicated GPU resources.

To interact with dedicated endpoints, provide the endpoint ID in the `model` argument.

```python
import os
from friendli import Friendli

client = Friendli(
    team_id=os.environ["TEAM_ID"],  # If not provided, default team is used.
    use_dedicated_endpoint=True,
)

chat_completion = client.chat.completions.create(
    model=os.environ["ENDPOINT_ID"],
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
)
print(chat_completion.choices[0].message.content)
```

# Friendli Container

[Friendli Container](https://friendli.ai/products/container) is perfect for users who prefer to serve LLMs within their own infrastructure. By deploying the Friendli Engine in containers on your on-premise or cloud GPUs, you can maintain complete control over your data and operations, ensuring security and compliance with internal policies.

```python
from friendli import Friendli

client = Friendli(base_url="http://0.0.0.0:8000")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
)
print(chat_completion.choices[0].message.content)
```

# Async Usage

```python
import asyncio
from friendli import AsyncFriendli

client = AsyncFriendli()

async def main() -> None:
    chat_completion = await client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[
            {
                "role": "user",
                "content": "Tell me how to make a delicious pancake",
            }
        ],
    )
    print(chat_completion.choices[0].message.content)


asyncio.run(main())
```

# Streaming Usage

```python
from friendli import Friendli

client = Friendli()

stream = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The async client (`AsyncFriendli`) uses the same interface to stream the response.

```python
import asyncio
from friendli import AsyncFriendli

client = AsyncFriendli()

async def main() -> None:
    stream = await client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[
            {
                "role": "user",
                "content": "Tell me how to make a delicious pancake",
            }
        ],
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


asyncio.run(main())
```

# Advanced Usage

## Sending Requests to LoRA Adapters

If your endpoint is serving a Multi-LoRA model, you can send request to one of the adapters by providing the adapter route in the `model` argument.

For Friendli Dedicated Endpoints, provide the endpoint ID and the adapter route separated by a colon (`:`).

```python
import os
from friendli import Friendli

client = Friendli(
    team_id=os.environ["TEAM_ID"],  # If not provided, default team is used.
    use_dedicated_endpoint=True,
)

chat_completion = client.lora.completions.create(
    model=f"{os.environ['ENDPOINT_ID']}:{os.environ['ADAPTER_ROUTE']}",
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
)
```

For Friendli Container, just provide the adapter name.

```python
import os
from friendli import Friendli

client = Friendli(base_url="http://0.0.0.0:8000")

chat_completion = client.lora.completions.create(
    model=os.environ["ADAPTER_NAME"],
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
)
```

## Using the gRPC Interface

> [!IMPORTANT]
> gRPC is only supported by Friendli Container, and only the streaming API of `v1/completions` is available.

When Frienldi Container is running in gPRC mode, the client can interact with the gRPC server
by initializing it with `use_grpc=True` argument.

```python
from friendli import Friendli

client = Friendli(base_url="0.0.0.0:8000", use_grpc=True)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
    stream=True,  # Only streaming mode is available
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Configuring the HTTP Client

The client uses `httpx` to send HTTP requests. You can provide the customized `httpx.Client` when initializing `Friendli`.

```python
import httpx
from friendli import Friendli

with httpx.Client() as client:
    client = Friendli(http_client=http_client)
```

For the async client, you can provide `httpx.AsyncClient`.

```python
import httx
from friendli import AsyncFriendli

with httpx.AsyncClient() as client:
    client = AsyncFriendli(http_client=http_client)
```

## Configuring the gRPC Channel

```python
import grpc
from friendli import Friendli

with grpc.insecure_channel("0.0.0.0:8000") as channel:
    client = Friendli(use_grpc=True, grpc_channel=channel)
```

You can use the same interface for the async client.

```python
import grpc.aio
from friendli import AsyncFriendli

async with grpc.aio.insecure_channel("0.0.0.0:8000") as channel:
    client = AsyncFriendli(use_grpc=True, grpc_channel=channel)
```

## Managing Resource

The Friendli client provides several methods to manage and release resources.

### Closing the Client

Both the `Friendli` and `AsyncFriendli` clients can hold network connections or other resources during their lifetime.
To ensure these resources are properly released, you should either call the `close()` method or use the client within a context manager.

```python
from friendli import Friendli

client = Friendli()

# Use the client for various operations...

# When done, close the client to release resources
client.close()
```

For the asynchronous client, the pattern is similar:

```python
import asyncio
from friendli import AsyncFriendli

client = AsyncFriendli()

# Use the client for various async operations...

# When done, close the client to release resources
await client.close()
```

You can also use context manager to automatically close the client and releases resources when the block is exited, making it a safer and more convenient way to manage resources.

```python
from friendli import Friendli

with Friendli() as client:
    ...
```

For asynchronous usage:

```python
import asyncio
from friendli import AsyncFriendli

async def main():
    async with AsyncFriendli() as client:
        ...


asyncio.run(main())
```

### Managing Streaming Responses

When using streaming responses, it’s crucial to properly close the HTTP connection after the interaction is complete.
By default, the connection is automatically closed once all data from the stream has been consumed (i.e., when the for-loop reaches the end).
However, if streaming is interrupted by exceptions or other issues, the connection may remain open and won’t be released until it is garbage-collected.
To ensure that all underlying connections and resources are properly released, it’s important to explicitly close the connection, particularly when streaming is prematurely terminated.

```python
from friendli import Friendli

client = Friendli()

stream = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
    stream=True,
)

try:
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
finally:
    stream.close()  # Ensure the stream is closed after use
```

For asynchronous streaming:

```python
import asyncio
from friendli import AsyncFriendli

client = AsyncFriendli()

async def main():
    stream = await client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[
            {
                "role": "user",
                "content": "Tell me how to make a delicious pancake",
            }
        ],
        stream=True,
    )

    try:
        async for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
    finally:
        await stream.close()  # Ensure the stream is closed after use

asyncio.run(main())
```

You can also use context manager to automatically close the client and releases resources when the block is exited, making it a safer and more convenient way to manage resources.

```python
from friendli import Friendli

client = Friendli()

with client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
    stream=True,
) as stream:
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

For asynchronous streaming:

```python
import asyncio
from friendli import AsyncFriendli

client = AsyncFriendli()

async def main():
    async with await client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[
            {
                "role": "user",
                "content": "Tell me how to make a delicious pancake",
            }
        ],
        stream=True,
    ) as stream:
        async for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="", flush=True)

asyncio.run(main())
```

### Canceling a gRPC Stream

When using the gRPC interface with streaming, you might want to cancel an ongoing stream operation before it completes. This is particularly useful if you need to stop the stream due to a timeout or some other condition.

For synchronous gRPC streaming:

```python
from friendli import Friendli

client = Friendli(base_url="0.0.0.0:8000", use_grpc=True)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake",
        }
    ],
    stream=True,
)

try:
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
except SomeException:
    stream.cancel()  # Cancel the stream in case of an error or interruption
```

For asynchronous gRPC streaming:

```python
import asyncio
from friendli import AsyncFriendli

client = AsyncFriendli(base_url="0.0.0.0:8000", use_grpc=True)

async def main():
    stream = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Tell me how to make a delicious pancake",
            }
        ],
        stream=True,
    )

    try:
        async for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
    except SomeException:
        stream.cancel()  # Cancel the stream in case of an error or interruption

asyncio.run(main())
```

# CLI Examples

You can also call the generation APIs directly with CLI.

```sh
friendli api chat-completions create \
  -g "user Tell me how to make a delicious pancake" \
  -m meta-llama-3.1-8b-instruct
```

For further information about the `friendli` command, run `friendli --help` in your terminal shell.
This will provide you with a detailed list of available options and usage instructions.

> [!TIP] > **Check out our [official documentation](https://friendli.ai/docs/) to learn more!**
