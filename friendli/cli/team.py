# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Organization CLI."""

from __future__ import annotations

import typer

from friendli.client.graphql.user import UserGqlClient
from friendli.context import get_current_team_id, set_current_team_id
from friendli.formatter import TableFormatter
from friendli.utils.decorator import check_api
from friendli.utils.format import secho_error_and_exit

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
)

team_table_formatter = TableFormatter(
    name="Teams",
    fields=["node.id", "node.name", "node.dedicated.plan"],
    headers=["ID", "Name", "Dedicated Plan"],
    substitute_exact_match_only=False,
)


@app.command("list")
@check_api
def list_teams():
    """List teams."""
    client = UserGqlClient()
    teams = client.get_teams()
    current_team_id = get_current_team_id()

    for team in teams:
        if current_team_id is not None and team["node"]["id"] == current_team_id:
            team["node"]["id"] = f"[bold green]* {team['node']['id']}"
            team["node"]["name"] = f"[bold green]{team['node']['name']}"
            if team["node"]["dedicated"] is not None:
                team["node"]["dedicated"][
                    "plan"
                ] = f"[bold green]{team['node']['dedicated']['plan']}"
            else:
                team["node"]["dedicated"] = {"plan": "[bold green]-"}
        else:
            team["node"]["id"] = f"  {team['node']['id']}"

    team_table_formatter.render(teams)


@app.command("switch")
def switch_team(team_id: str = typer.Argument(..., help="ID of team to switch.")):
    """Switch current team context to run as."""
    client = UserGqlClient()
    accessible_team_ids = client.get_team_ids()
    if team_id not in accessible_team_ids:
        secho_error_and_exit(f"'{team_id}' is not valid team ID.")

    set_current_team_id(team_id)
    typer.secho(
        f"Team context is switched to '{team_id}'.",
        fg=typer.colors.GREEN,
    )
