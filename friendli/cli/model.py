# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=redefined-builtin, too-many-locals, too-many-arguments, line-too-long

"""Friendli Model CLI."""

from __future__ import annotations

import typer

from friendli.formatter import TableFormatter
from friendli.sdk.client import Friendli
from friendli.utils.compat import model_dump
from friendli.utils.decorator import check_api

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
)

table_formatter = TableFormatter(
    name="Models",
    fields=[
        "id",
        "name",
    ],
    headers=[
        "ID",
        "Name",
    ],
)


@app.command("list")
@check_api
def list_models():
    """List models."""
    client = Friendli()
    models = client.model.list()
    models_ = [model_dump(model) for model in iter(models)]
    table_formatter.render(models_)
