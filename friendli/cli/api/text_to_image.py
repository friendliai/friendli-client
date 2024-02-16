# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-many-arguments

"""Generates image from text via CLI."""

from __future__ import annotations

from typing import Optional

import typer

from friendli.enums import ResponseFormat
from friendli.sdk.client import Friendli
from friendli.utils.compat import model_dump
from friendli.utils.decorator import check_api

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
)


@app.command()
@check_api
def create(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help=(
            "The model to use for chat completions. "
            "See https://docs.friendli.ai/guides/serverless_endpoints/pricing for more "
            "about available models and pricing."
        ),
    ),
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="A text description of the desired image(s).",
    ),
    negative_prompt: Optional[str] = typer.Option(
        None,
        "--negative-prompt",
        "-N",
        help="A text specifying what you don't want in your image(s).",
    ),
    num_outputs: Optional[int] = typer.Option(
        None,
        "--num-outputs",
        "-n",
        min=1,
        max=16,
        help="The number of images to generate. Only 1 output will be generated when not provided.",
    ),
    num_inference_steps: Optional[int] = typer.Option(
        None,
        "--num-inference-steps",
        "-I",
        min=1,
        max=500,
        help=(
            "The number of inference steps for denoising process. 50 steps will be "
            "taken when not provided."
        ),
    ),
    guidance_scale: Optional[float] = typer.Option(
        None,
        "--guidance-scale",
        "-G",
        help=(
            "Guidance scale to control how much generation process adheres to the text "
            "prompt. When not provided, it is set to 7.5."
        ),
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Seed to control random procedure.",
    ),
    response_format: ResponseFormat = typer.Option(
        ResponseFormat.URL,
        "--response-format",
        "-R",
        help="The format in which the generated images are returned.",
    ),
    token: Optional[str] = typer.Option(
        None, "--token", "-t", help="Personal access token for auth."
    ),
    team_id: Optional[str] = typer.Option(
        None, "--team-id", help="ID of team to run as."
    ),
):
    """Create image from text."""
    client = Friendli(token=token, team_id=team_id)

    image = client.images.text_to_image.create(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_outputs=num_outputs,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        response_format=response_format,
    )
    typer.echo(model_dump(image))
