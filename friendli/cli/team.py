# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Organization CLI."""

from __future__ import annotations

import typer

from friendli.client.graphql.base import get_default_gql_client
from friendli.client.graphql.user import UserGqlClient
from friendli.context import get_current_team_id, set_current_team_id
from friendli.formatter import PanelFormatter, TableFormatter
from friendli.utils.decorator import check_api

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
)

team_table_formatter = TableFormatter(
    name="Teams", fields=["node.id", "node.name"], headers=["ID", "Name"]
)
team_panel_formatter = PanelFormatter(
    name="Team Detail",
    fields=["id", "name", "state"],
    headers=["ID", "Name", "State"],
)
member_table_formatter = TableFormatter(
    name="Members",
    fields=["id", "name", "email", "privilege_level"],
    headers=["ID", "Name", "Email", "Role"],
)


@app.command("list")
@check_api
def list_teams():
    """List teams."""
    gql_client = get_default_gql_client()
    client = UserGqlClient(client=gql_client)
    teams = client.get_teams()
    current_team_id = get_current_team_id()

    for team in teams:
        if current_team_id is not None and team["node"]["id"] == current_team_id:
            team["node"]["id"] = f"[bold green]* {team['node']['id']}"
            team["node"]["name"] = f"[bold green]{team['node']['name']}"
        else:
            team["node"]["id"] = f"  {team['node']['id']}"

    team_table_formatter.render(teams)


@app.command("switch")
def switch_team(team_id: str = typer.Argument(..., help="ID of team to switch.")):
    """Switch current team context to run as."""
    set_current_team_id(team_id)
