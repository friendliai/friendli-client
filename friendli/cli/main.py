# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli CLI."""

# pylint: disable=line-too-long

from __future__ import annotations

import typer

import friendli
from friendli.auth import TokenType, clear_tokens, update_token
from friendli.cli import api, checkpoint
from friendli.cli.login import oauth2_login, pwd_login
from friendli.client.graphql.base import get_default_gql_client
from friendli.client.graphql.team import TeamGqlClient
from friendli.client.graphql.user import UserGqlClient
from friendli.context import (
    clear_contexts,
    get_current_project_id,
    get_current_team_id,
    set_current_project_id,
    set_current_team_id,
)
from friendli.errors import AuthTokenNotFoundError
from friendli.formatter import PanelFormatter
from friendli.utils.decorator import check_api
from friendli.utils.format import secho_error_and_exit
from friendli.utils.version import get_installed_version

app = typer.Typer(
    help="Supercharge Generative AI Serving 🚀",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
    pretty_exceptions_enable=False,
)

# app.add_typer(deployment.app, name="deployment", help="Manage deployments")
# app.add_typer(project.app, name="project", help="Manage projects")
# app.add_typer(team.app, name="team", help="Manage teams")
app.add_typer(checkpoint.app, name="checkpoint", help="Manage checkpoints")
app.add_typer(api.app, name="api", help="API call to endpoints")


user_panel_formatter = PanelFormatter(
    name="My Info",
    fields=["name", "email"],
    headers=["Name", "Email"],
)


# @app.command()
@check_api
def whoami():
    """Show my user info."""
    try:
        gql_client = get_default_gql_client()
        client = UserGqlClient(client=gql_client)
        info = client.get_current_user_info()
    except AuthTokenNotFoundError as exc:
        secho_error_and_exit(str(exc))

    user_panel_formatter.render([info])


# @app.command()
@check_api
def login(
    use_sso: bool = typer.Option(False, "--sso", help="Use SSO login."),
):
    """Sign in Friendli."""
    if friendli.token:
        typer.secho(
            "You've already set the 'FRIENDLI_TOKEN' environment variable for "
            "authentication, which takes precedence over the login session. Using both "
            "authentication methods could lead to unexpected issues. We suggest "
            "unsetting the 'FRIENDLI_TOKEN' environment variable if you prefer to log "
            "in through the standard login session.",
            fg=typer.colors.RED,
        )

    if use_sso:
        access_token, refresh_token = oauth2_login()
    else:
        email = typer.prompt("Enter your email")
        pwd = typer.prompt("Enter your password", hide_input=True)
        access_token, refresh_token = pwd_login(email, pwd)

    update_token(token_type=TokenType.ACCESS, token=access_token)
    update_token(token_type=TokenType.REFRESH, token=refresh_token)

    # Set default team and project.
    team_id = _set_default_team()
    if team_id is not None:
        _set_default_project(team_id)

    _display_login_success()


# @app.command()
def logout():
    """Sign out."""
    clear_tokens()
    clear_contexts()
    typer.secho("Successfully signed out.", fg=typer.colors.BLUE)


@app.command()
def version():
    """Check the installed package version."""
    installed_version = get_installed_version()
    typer.echo(installed_version)


def _display_login_success():
    typography = r"""
 _____                         _  _
|  ___|_  _(_) ___  _  __    _| || |(_)  
| |__ | '__| |/ _ \| '__  \/ _  || || |
|  __|| |  | |  __/| |  | | (_) || || |
|_|   |_|  |_|\___||_|  |_|\___/ |_||_|
"""
    typer.secho(f"\nLOGIN SUCCESS!\n{typography}", fg=typer.colors.BLUE)


def _set_default_team() -> str | None:
    current_team_id = get_current_team_id()
    if current_team_id is None:
        gql_client = get_default_gql_client()
        client = UserGqlClient(client=gql_client)
        teams = client.get_teams()
        if len(teams) > 0:
            default_team_id = teams[0]["node"]["id"]
            set_current_team_id(default_team_id)
            typer.secho(
                f"📌 CLI configures your team to '{default_team_id}'. If you want to "
                "switch the team, run 'friendli team switch'.",
                fg=typer.colors.YELLOW,
            )
            return default_team_id

        typer.secho(
            "Team authorized to access is not found. Contact to your team admin.",
            fg=typer.colors.RED,
        )
        return None

    return current_team_id


def _set_default_project(team_id: str) -> str | None:
    current_project_id = get_current_project_id()
    if current_project_id is None:
        gql_client = get_default_gql_client()
        client = TeamGqlClient(client=gql_client)
        projects = client.get_projects(team_id=team_id)
        if len(projects) > 0:
            default_project_id = projects[0]["node"]["id"]
            set_current_project_id(default_project_id)
            typer.secho(
                f"📌 CLI configures your project to '{default_project_id}'. If you want "
                "to switch the project, run 'friendli project switch'.",
                fg=typer.colors.YELLOW,
            )
            return default_project_id

        typer.secho(
            "Project authorized to access is not found. Contact to your team admin.",
            fg=typer.colors.RED,
        )
        return None

    return current_project_id
