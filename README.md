<!---
Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.
-->

<p align="center">
  <img src="https://docs.friendli.ai/img/logo.svg" width="30%" alt="Friendli Logo">
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
  <a href="https://docs.friendli.ai/">
    <img alt="Documentation" src="https://img.shields.io/badge/read-doc-blue?logo=ReadMe&logoColor=white">
  </a>
  <a href="https://github.com/friendliai/friendli-client/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=Apache">
  </a>
</p>

Welcome to Friendli Suite, the ultimate solution for serving generative AI models. We offer three distinct options to cater to your specific needs, each designed to provide superior performance, cost-effectiveness, and ease of use.

# Friendli Suite

## 1. Friendli Serverless Endpoints

Imagine a playground for your AI dreams.
Friendli Serverless Endpoint is just that - a simple, click-and-play interface that lets you access popular open-source models like Llama-2 and Stable Diffusion without any heavy lifting.
Choose your model, enter your prompt or upload an image, and marvel at the generated text, code, or image outputs.
With pay-per-token billing, this is ideal for exploration and experimentation.
You can think of it as an AI sampler.

## 2. Friendli Dedicated Endpoints

Ready to take the reins and unleash the full potential of your own models?
Friendli Dedicated Endpoint is for you.
This service provides dedicated GPU resources in the cloud platform of your choice (AWS, GCP, Azure), letting you upload and run your custom generative AI models.
Reserve the exact GPU you need (A10, A100 40G, A100 80G, etc.) and enjoy fine-grained control over your model settings.
Pay-per-second billing makes it perfect for regular or resource-intensive workloads.

## 3. Friendli Container

Do you prefer the comfort and security of your own data center?
Friendli Container is the solution.
We provide the Friendli Engine within Docker containers that can be installed on your on-premise GPUs so your data stays within your own secure cluster.
This option offers maximum control and security, ideal for advanced users or those with specific data privacy requirements.

> [!NOTE]
>
> ## The Friendli Engine: The Powerhouse Behind the Suite
>
> At the heart of each Friendli Suite offering lies the Friendli Engine, a patented, GPU-optimized serving engine.
> This technological marvel is what enables Friendli Suite's superior performance and cost-effectiveness, featuring innovations like continuous batching (iteration batching) that significantly improve resource utilization compared to traditional LLM serving solutions.

# 🕹️ Friendli Client

## Installation

```sh
pip install friendli-client
```

> [!NOTE]
> If you have a Hugging Face checkpoint and want to convert it to a Friendli-compatible format with applying quantization, you need to install the package with the necessary machine learing library (`mllib`) dependencies.
> In this case, install the package with the following command:
>
> ```sh
> pip install "friendli-client[mllib]"
> ```

## Python SDK Examples

> [!IMPORTANT]
> You must set `FRIENDLI_TOKEN` environment variable before initializing the client instance with `client = Friendli()`.
> Alternatively, you can provide the value of your personal access token as the `token` argument when creating the client, like so:
>
> ```python
> from friendli import Friendli
> 
> client = Friendli(token="YOUR PERSONAL ACCESS TOKEN")
> ```

### Default

```python
from friendli import Friendli

client = Friendli()

chat_completion = client.chat.completions.create(
    model="llama-2-13b-chat",
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake"
        }
    ],
    stream=False,
)
print(chat_completion.choices[0].message.content)
```

### Streaming

```python
from friendli import Friendli

client = Friendli()

stream = client.chat.completions.create(
    model="llama-2-13b-chat",
    messages=[
        {
            "role": "user",
            "content": "Tell me how to make a delicious pancake"
        }
    ]
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Async

```python
import asyncio
from friendi import AsyncFriendli

client = AsyncFriendli()


async def main() -> None:
    chat_completion = await client.chat.completions.create(
        model="llama-2-13b-chat",
        messages=[
            {
                "role": "user",
                "content": "Tell me how to make a delicious pancake"
            }
        ]
        stream=False,
    )
    print(chat_completion.choices[0].message.content)


asyncio.run(main())
```

### Streaming (Async)

```python
import asyncio
from friendi import AsyncFriendli

client = AsyncFriendli()


async def main() -> None:
    stream = await client.chat.completions.create(
        model="llama-2-13b-chat",
        messages=[
            {
                "role": "user",
                "content": "Tell me how to make a delicious pancake"
            }
        ]
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")


asyncio.run(main())
```

## CLI Examples

You can also call the generation APIs directly with CLI.

```sh
friendli api chat-completions create \
  -g "user Tell me how to make a delicious pancake" \
  -m llama-2-13b-chat
```

For further information about the `friendli` command, run `friendli --help` in your terminal shell.
This will provide you with a detailed list of available options and usage instructions.

> [!TIP]
> **Check out our [official documentations](https://docs.periflow.ai/) to learn more!**
