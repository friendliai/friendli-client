# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# pylint: disable=redefined-builtin

"""Friendli Catalog CLI."""

from __future__ import annotations

import typer

from friendli.formatter import (
    JSONFormatter,
    PanelFormatter,
    TableFormatter,
    TreeFormatter,
)
from friendli.sdk.resource.catalog import Catalog
from friendli.utils.decorator import check_api
from friendli.utils.format import datetime_to_pretty_str, secho_error_and_exit

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)

table_formatter = TableFormatter(
    name="Catalog",
    fields=[
        "name",
        "use_count",
        "tags",
        "created_at",
    ],
    headers=[
        "Name",
        "# Uses",
        "Tags",
        "Created At",
    ],
)
panel_formatter = PanelFormatter(
    name="Overview",
    fields=[
        "id",
        "name",
        "summary",
        "description",
        "use_count",
        "created_at",
        "tags",
    ],
    headers=[
        "ID",
        "Name",
        "Summary",
        "Description",
        "# Uses",
        "Created At",
        "Tags",
    ],
)
json_formatter = JSONFormatter(name="Attributes")
tree_formatter = TreeFormatter(name="Files")


@app.command()
@check_api
def list(
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="The name of public checkpoint to search.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="The number of public checkpoints to display.",
    ),
):
    """Lists public checkpoints in catalog."""
    client = Catalog()
    catalogs = client.list(name=name, limit=limit)

    catalog_dicts = []
    for catalog in catalogs:
        catalog_dict = catalog.model_dump()
        catalog_dict["created_at"] = datetime_to_pretty_str(catalog.created_at)
        catalog_dicts.append(catalog_dict)

    table_formatter.render(catalog_dicts)


@app.command()
@check_api
def view(
    name: str = typer.Argument(),
):
    """Displays info of a catalog."""
    client = Catalog()
    catalogs = client.list(name=name)

    catalog = None
    for cat in catalogs:
        if cat.name == name:
            catalog = cat
    if catalog is None:
        msg = f"Public checkpoint with name '{name}' is not found in the catalog."
        if len(catalogs) > 0:
            msg += f" Did you mean '{catalogs[0].name}'?"
        secho_error_and_exit(msg)

    catalog_dict = catalog.model_dump()
    catalog_dict["created_at"] = datetime_to_pretty_str(catalog.created_at)

    panel_formatter.render([catalog_dict])
    json_formatter.render(catalog_dict["attributes"])
    tree_formatter.render(catalog_dict["files"])
