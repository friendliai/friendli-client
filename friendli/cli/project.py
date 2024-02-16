# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Project CLI."""

from __future__ import annotations

import typer

from friendli.client.graphql.base import get_default_gql_client
from friendli.client.graphql.team import TeamGqlClient
from friendli.context import (
    get_current_project_id,
    get_current_team_id,
    set_current_project_id,
)
from friendli.formatter import PanelFormatter, TableFormatter
from friendli.utils.decorator import check_api
from friendli.utils.format import secho_error_and_exit

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
)
project_table_formatter = TableFormatter(
    name="Project", fields=["node.id", "node.name"], headers=["ID", "Name"]
)
project_panel_formatter = PanelFormatter(
    name="Project Detail",
    fields=["pf_group_id", "id", "name"],
    headers=["Organization ID", "Project ID", "Name"],
)
member_table_formatter = TableFormatter(
    name="Members",
    fields=["id", "name", "email", "access_level"],
    headers=["ID", "Name", "Email", "Role"],
)


# pylint: disable=redefined-builtin
@app.command()
@check_api
def list():
    """List all accessible projects."""
    gql_client = get_default_gql_client()
    client = TeamGqlClient(client=gql_client)

    team_id = get_current_team_id()
    if team_id is None:
        secho_error_and_exit(
            "Team is not configured. Set 'FRIENDLI_TEAM' environment variable, or use "
            "a CLI command 'friendli team switch'."
        )
    projects = client.get_projects(team_id=team_id)
    current_project_id = get_current_project_id()

    for project in projects:
        if (
            current_project_id is not None
            and project["node"]["id"] == current_project_id
        ):
            project["node"]["id"] = f"[bold green]* {project['node']['id']}"
            project["node"]["name"] = f"[bold green]{project['node']['name']}"
        else:
            project["node"]["id"] = f"  {project['node']['id']}"

    project_table_formatter.render(projects)


@app.command()
@check_api
def switch(
    project_id: str = typer.Argument(
        ...,
        help="ID of project to switch.",
    )
):
    """Switch currnet project context to run as."""
    set_current_project_id(project_id)
