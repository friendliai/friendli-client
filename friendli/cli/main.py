# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli CLI."""

# pylint: disable=line-too-long

from __future__ import annotations

import requests
import typer
from requests import HTTPError, Response

from friendli.auth import TokenType, clear_tokens, get_token, update_token
from friendli.cli import api, checkpoint
from friendli.client.project import ProjectClient
from friendli.client.user import UserClient, UserGroupClient, UserMFAClient
from friendli.context import (
    get_current_project_id,
    project_context_path,
    set_current_group_id,
)
from friendli.di.injector import get_injector
from friendli.errors import AuthTokenNotFoundError
from friendli.formatter import PanelFormatter
from friendli.utils.decorator import check_api
from friendli.utils.format import secho_error_and_exit
from friendli.utils.request import DEFAULT_REQ_TIMEOUT
from friendli.utils.url import URLProvider
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
        client = UserClient()
        info = client.get_current_user_info()
    except AuthTokenNotFoundError as exc:
        secho_error_and_exit(str(exc))

    user_panel_formatter.render([info])


# @app.command()
@check_api
def login(
    email: str = typer.Option(..., prompt="Enter your email"),
    password: str = typer.Option(..., prompt="Enter your password", hide_input=True),
):
    """Sign in."""
    injector = get_injector()
    url_provider = injector.get(URLProvider)
    r = requests.post(
        url_provider.get_web_backend_uri("/api/auth/cli/access_token"),
        json={"username": email, "password": password},
        timeout=DEFAULT_REQ_TIMEOUT,
    )
    try:
        resp = r.json()
    except requests.exceptions.JSONDecodeError:
        if r.status_code != 200:
            secho_error_and_exit(r.content.decode())
        secho_error_and_exit("Invalid response format.")

    if "code" in resp and resp["code"] == "mfa_required":
        mfa_token = resp["mfaToken"]
        client = UserMFAClient()
        # TODO: MFA type currently defaults to totp, need changes when new options are added
        client.initiate_mfa(mfa_type="totp", mfa_token=mfa_token)
        update_token(token_type=TokenType.MFA, token=mfa_token)
        typer.run(_mfa_verify)
    else:
        _handle_login_response(r, False)

    # Save user's organiztion context
    project_client = ProjectClient()
    user_group_client = UserGroupClient()

    try:
        org = user_group_client.get_group_info()
    except IndexError:
        secho_error_and_exit("You are not included in any organization.")
    org_id = org["id"]

    project_id = get_current_project_id()
    if project_id is not None:
        if project_client.check_project_membership(pf_project_id=project_id):
            project_org_id = project_client.get_project(pf_project_id=project_id)[
                "pf_group_id"
            ]
            if project_org_id != org_id:
                project_context_path.unlink(missing_ok=True)
        else:
            project_context_path.unlink(missing_ok=True)
    set_current_group_id(org_id)


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


def _mfa_verify(_, code: str = typer.Option(..., prompt="Enter MFA Code")):
    injector = get_injector()
    url_provider = injector.get(URLProvider)

    mfa_token = get_token(TokenType.MFA)
    # TODO: MFA type currently defaults to totp, need changes when new options are added
    mfa_type = "totp"
    username = f"mfa://{mfa_type}/{mfa_token}"
    r = requests.post(
        url_provider.get_web_backend_uri("/api/auth/cli/access_token"),
        json={"username": username, "password": code},
        timeout=DEFAULT_REQ_TIMEOUT,
    )
    _handle_login_response(r, True)


def _handle_login_response(r: Response, mfa: bool):
    try:
        r.raise_for_status()
        update_token(token_type=TokenType.ACCESS, token=r.json()["accessToken"])
        update_token(token_type=TokenType.REFRESH, token=r.json()["refreshToken"])

        typer.echo("\n\nLogin success!")
        typer.echo("Welcome back to...")
        typography = r"""
 _____                         _  _
|  ___|_  _(_) ___  _  __    _| || |(_)  
| |__ | '__| |/ _ \| '__  \/ _  || || |
|  __|| |  | |  __/| |  | | (_) || || |
|_|   |_|  |_|\___||_|  |_|\___/ |_||_|
"""
        typer.secho(typography, fg=typer.colors.BLUE)
    except HTTPError:
        if mfa:
            secho_error_and_exit("Login failed... Invalid MFA Code.")
        else:
            secho_error_and_exit(
                "Login failed... Please check your email and password."
            )
