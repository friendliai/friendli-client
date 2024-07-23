# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Init."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from typer import Argument, Option, Typer

from ..const import Panel
from ..context import AppContext, TyperAppContext
from ..typer_util import CommandUsageExample, format_examples

app = Typer(
    no_args_is_help=True,
    name="model",
    rich_help_panel=Panel.DEDICATED,
    help="Manage your Dedicated Models.",
)


@app.command(
    "list",
    help="""
List models.
""",
    epilog=format_examples(
        [
            CommandUsageExample(
                synopsis=("List models in a project. [yellow](RECOMMENDED)[/]"),
                args="friendli model list --project PROJECT_ID",
            ),
        ]
    ),
)
def _list(
    ctx: TyperAppContext,
    project: str = Option("--project", help="Project ID"),
) -> None:
    from ..action.dedicated_model_list import run

    with AppContext(ctx.obj) as app_ctx:
        run(app_ctx, project)


@app.command(
    "push",
    help="Upload a model to Friendli Dedicated Endpoints.",
    epilog=format_examples(
        [
            CommandUsageExample(
                synopsis=("Upload a base model. [yellow](RECOMMENDED)[/]"),
                args="friendli model push",
            ),
            CommandUsageExample(
                synopsis=("Upload an adapter model. [yellow](RECOMMENDED)[/]"),
                args="friendli model push --base MODEL_ID",
            ),
            CommandUsageExample(
                synopsis=("Choose different directory to upload from."),
                args="friendli model push ./model",
            ),
            CommandUsageExample(
                synopsis=("Specify the model name."),
                args="friendli model push --name MODEL_NAME",
            ),
            CommandUsageExample(
                synopsis=("Specify the project to upload the model to."),
                args="friendli model push --project PROJECT_ID",
            ),
        ]
    ),
)
def _push(
    ctx: TyperAppContext,
    model_path: Optional[Path] = Argument(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    base: Optional[str] = Option(None, "--base", help="Base model ID"),
    name: Optional[str] = Option(None, "--name", help="Model name"),
    project: Optional[str] = Option(None, "--project", help="Project ID"),
) -> None:
    model_path = model_path or Path.cwd()

    if base:
        from ..action.dedicated_model_push_adapter import run as _push_adapter

        with AppContext(ctx.obj) as app_ctx:
            _push_adapter(app_ctx, model_path, base, name, project)

        return

    from ..action.dedicated_model_push_base import run as _push_base

    with AppContext(ctx.obj) as app_ctx:
        _push_base(app_ctx, model_path, name, project)
