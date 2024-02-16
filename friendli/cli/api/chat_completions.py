# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-many-locals, too-many-arguments

"""Generates Chat Completions via CLI."""

from __future__ import annotations

from typing import List, Optional

import typer

from friendli.schema.api.v1.chat.completions import MessageParam
from friendli.sdk.client import Friendli
from friendli.utils.compat import model_dump
from friendli.utils.decorator import check_api
from friendli.utils.format import secho_error_and_exit

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
)


@app.command()
@check_api
def create(
    messages: List[str] = typer.Option(
        ...,
        "--message",
        "-g",
        help=(
            "A message in `ROLE CONTENT` format. Repeat this option to add multiple messages."
        ),
    ),
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        case_sensitive=True,
        help=(
            "The model to use for chat completions. "
            "See https://docs.friendli.ai/guides/serverless_endpoints/pricing for more "
            "about available models and pricing."
        ),
    ),
    n: Optional[int] = typer.Option(
        None,
        "--n",
        "-n",
        min=1,
        help="The number of results to generate.",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        "-M",
        min=1,
        help="The maximum number of tokens to generate.",
    ),
    stop: Optional[List[str]] = typer.Option(
        None,
        "--stop",
        "-S",
        help=(
            "When one of the stop phrases appears in the generation result, the API "
            "will stop generation. The stop phrases are excluded from the result. "
            "Repeat this option to use multiple stop phrases."
        ),
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        "-T",
        min=0,
        help="Sampling temperature. non-zero positive numbers are allowed.",
    ),
    top_p: Optional[float] = typer.Option(
        None,
        "--top-p",
        "-P",
        help="Tokens comprising the top top_p probability mass are kept for sampling.",
        min=0,
        max=1,
    ),
    frequency_penalty: Optional[float] = typer.Option(
        None,
        "--frequency-penalty",
        "-fp",
        min=-2,
        max=2,
        help=(
            "Positive values penalizes tokens that have been sampled, taking into "
            "account their frequency in the preceding text. This penalization "
            "diminishes the model's tendency to reproduce identical lines verbatim."
        ),
    ),
    presence_penalty: Optional[float] = typer.Option(
        None,
        "--presence-penalty",
        "-pp",
        min=-2.0,
        max=2.0,
        help=(
            "Positive values penalizes tokens that have been sampled at least once in "
            "the existing text."
        ),
    ),
    enable_stream: bool = typer.Option(
        False,
        "--stream",
        "-s",
        help="Whether to stream generation result.",
    ),
    token: Optional[str] = typer.Option(
        None, "--token", "-t", help="Personal access token for auth."
    ),
    team_id: Optional[str] = typer.Option(
        None, "--team-id", help="ID of team to run as."
    ),
):
    """Creates chat completions."""
    client = Friendli(token=token, team_id=team_id)
    if enable_stream:
        stream = client.chat.completions.create(
            stream=True,
            model=model,
            messages=_prepare_messages(messages),
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )
        for chunk in stream:
            if n is not None and n > 1:
                typer.echo(model_dump(chunk))
            else:
                typer.echo(chunk.choices[0].delta.content or "", nl=False)
    else:
        chat_completion = client.chat.completions.create(
            stream=False,
            model=model,
            messages=_prepare_messages(messages),
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )
        typer.echo(model_dump(chat_completion))


def _prepare_messages(messages: List[str]) -> List[MessageParam]:
    processed_messages: List[MessageParam] = []
    for message in messages:
        role, content = message.split(" ", 1)
        if role not in ("user", "assistant", "system"):
            secho_error_and_exit("The role should be one of {user, assistant, system}.")

        processed_message: MessageParam = {
            "role": role,
            "content": content,
        }
        processed_messages.append(processed_message)
    return processed_messages
