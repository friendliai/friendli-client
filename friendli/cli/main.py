# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli CLI."""

# pylint: disable=line-too-long

from __future__ import annotations

import typer

import friendli
from friendli.auth import TokenType, clear_tokens, update_token
from friendli.cli import api, checkpoint
from friendli.cli.login import oauth2_login, pwd_login
from friendli.client.user import UserClient
from friendli.errors import AuthTokenNotFoundError
from friendli.formatter import PanelFormatter
from friendli.graphql.user import UserGqlClient
from friendli.utils.decorator import check_api
from friendli.utils.format import secho_error_and_exit
from friendli.utils.version import get_installed_version

app = typer.Typer(
    help="Supercharge Generative AI Serving 🚀",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    pretty_exceptions_enable=False,
)

# Cloud-related features are temporarily disabled until account system is integrated with Friendli suite.
# app.add_typer(catalog.app, name="catalog", help="Manage catalog")
# app.add_typer(credential.app, name="credential", help="Manage credentials")
# app.add_typer(gpu.app, name="gpu", help="Manage GPUs")
# app.add_typer(deployment.app, name="deployment", help="Manage deployments")
# app.add_typer(project.app, name="project", help="Manage projects")
# app.add_typer(group.app, name="org", help="Manage organizations")
# app.add_typer(key.app, name="key", help="Manage api keys")
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
        client = UserGqlClient()
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
            "methods of authentication simultaneously could lead to unexpected issues. "
            "We suggest removing the 'FRIENDLI_TOKEN' environment variable if you "
            "prefer to log in through the standard login session.",
            fg=typer.colors.RED,
        )

    if use_sso:
        access_token, refresh_token = oauth2_login()
    else:
        email = typer.prompt("Enter your email")
        pwd = typer.prompt("Enter your password", hide_input=True)
        access_token, refresh_token = pwd_login(email, pwd)

    _display_login_success(access_token, refresh_token)


# @app.command()
def logout():
    """Sign out."""
    clear_tokens()
    typer.secho("Successfully signed out.", fg=typer.colors.BLUE)


# @app.command()
@check_api
def passwd(
    old_password: str = typer.Option(
        ..., prompt="Enter your current password", hide_input=True
    ),
    new_password: str = typer.Option(
        ..., prompt="Enter your new password", hide_input=True
    ),
    confirm_password: str = typer.Option(
        ..., prompt="Enter the new password again (confirmation)", hide_input=True
    ),
):
    """Change password."""
    if old_password == new_password:
        secho_error_and_exit("The current password is the same with the new password.")
    if new_password != confirm_password:
        secho_error_and_exit("Passwords did not match.")
    try:
        client = UserClient()
        client.change_password(old_password, new_password)
    except AuthTokenNotFoundError as exc:
        secho_error_and_exit(str(exc))

    typer.secho("Password is changed successfully!", fg=typer.colors.BLUE)


@app.command()
def version():
    """Check the installed package version."""
    installed_version = get_installed_version()
    typer.echo(installed_version)


def _display_login_success(access_token: str, refresh_token: str):
    update_token(token_type=TokenType.ACCESS, token=access_token)
    update_token(token_type=TokenType.REFRESH, token=refresh_token)

    typography = r"""
 _____                         _  _
|  ___|_  _(_) ___  _  __    _| || |(_)  
| |__ | '__| |/ _ \| '__  \/ _  || || |
|  __|| |  | |  __/| |  | | (_) || || |
|_|   |_|  |_|\___||_|  |_|\___/ |_||_|
"""
    typer.secho(f"\nLOGIN SUCCESS!\n{typography}", fg=typer.colors.BLUE)
