# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Catalog CLI."""

from __future__ import annotations

import typer

from periflow.formatter import TableFormatter
from periflow.sdk.resource.catalog import Catalog as CatalogAPI
from periflow.utils.format import datetime_to_pretty_str

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)

table_formatter = TableFormatter(
    name="Catalog",
    fields=[
        "id",
        "name",
        "use_count",
    ],
    headers=[
        "ID",
        "Name",
        "# Uses",
    ],
)


@app.command()
def list(
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="The name of publich checkpoint to search.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="The number of public checkpoints to display.",
    ),
):
    """List public checkpoints in catalog."""
    catalogs = CatalogAPI.list(name=name, limit=limit)
    catalog_dicts = []
    for catalog in catalogs:
        catalog_dict = catalog.model_dump()
        catalog_dict["created_at"] = datetime_to_pretty_str(catalog.created_at)
        catalog_dicts.append(catalog_dict)

    table_formatter.render(catalog_dicts)
