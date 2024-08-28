# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Generates with Friendli Serverless Endpoints APIs."""

from __future__ import annotations

import typer

from friendli.cli.api import chat_completions, completions

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
)

app.add_typer(chat_completions.app, name="chat-completions", help="Chat completions.")
app.add_typer(completions.app, name="completions", help="Text completions.")
