# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Organization CLI."""

from __future__ import annotations

from typing import Any, Dict
from uuid import UUID

import typer

from periflow.client.group import GroupClient
from periflow.client.user import UserClient, UserGroupClient
from periflow.context import get_current_group_id
from periflow.enums import GroupRole
from periflow.formatter import PanelFormatter, TableFormatter
from periflow.utils.format import secho_error_and_exit

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)

org_table_formatter = TableFormatter(
    name="Organization", fields=["name", "id"], headers=["Name", "ID"]
)
org_panel_formatter = PanelFormatter(
    name="Organization Detail",
    fields=["id", "name", "status"],
    headers=["ID", "Name", "Status"],
)
member_table_formatter = TableFormatter(
    name="Members",
    fields=["id", "username", "name", "email", "privilege_level"],
    headers=["ID", "Username", "Name", "Email", "Role"],
)


@app.command()
def invite(email: str = typer.Argument(..., help="Invitation recipient email address")):
    """Invite a new member to the organization.

    :::info
    Only the organization **Owner** can invite members.
    The invitation email will be sent to the provided email address.
    :::
    """
    group_client = GroupClient()

    org = get_current_org()

    if org["privilege_level"] != "owner":
        secho_error_and_exit("Only the owner of the organization can invite/set-role.")

    group_client.invite_to_group(org["id"], email)
    typer.echo("Invitation Successfully Sent!")


@app.command("accept-invite")
def accept_invite(
    token: str = typer.Option(..., prompt="Enter email token"),
    key: str = typer.Option(..., prompt="Enter verification key"),
):
    """Accept organization invitation.

    When the invitee accepts the invitation, the invitee becomes a **Member** of the invited
    organization.

    """
    group_client = GroupClient()
    group_client.accept_invite(token, key)
    typer.echo("Verification Success!")
    typer.echo("Please login again with: ", nl=False)
    typer.secho("pf login", fg=typer.colors.BLUE)


@app.command("set-role")
def set_role(
    username: str = typer.Argument(..., help="Username to set role"),
    role: GroupRole = typer.Argument(..., help="Organization role"),
):
    """Set organization role of the user.

    The organization-level roles and privileges are as follows:

    |                          | Owner | Member |
    |--------------------------|-------|--------|
    | Collaborate with teams   | ✓     | ✓      |
    | Invite members           | ✓     | ✗      |
    | Assign roles             | ✓     | ✗      |
    | Create & delete projects | ✓     | ✗      |
    | Manage payments          | ✓     | ✗      |
    | Delete organization      | ✓     | ✗      |

    """
    user_client = UserClient()

    org = get_current_org()

    if org["privilege_level"] != "owner":
        secho_error_and_exit(
            "Only the owner of the organization can invite/set-privilege."
        )

    user_id = _get_org_user_id_by_name(org["id"], username)
    user_client.set_group_privilege(org["id"], user_id, role)
    typer.echo(
        f"Organization role for user ({username}) successfully updated to {role.value}!"
    )


@app.command("delete-user")
def delete_user(
    username: str = typer.Argument(
        ...,
        help="Username to delete from the organization",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Forcefully delete without confirmation prompt",
    ),
):
    """Remove a user from the organization.

    :::info
    Only the **Owner** of the organization can remove a member.
    :::
    """
    user_client = UserClient()

    org = get_current_org()

    if org["privilege_level"] != "owner":
        secho_error_and_exit(
            "Only the owner of the organization can invite/set-privilege."
        )

    org_id = org["id"]
    user_id = _get_org_user_id_by_name(org_id, username)

    if not force:
        do_delete = typer.confirm(
            f"Are you sure to remove user({username}) from the organization?"
        )
        if not do_delete:
            raise typer.Abort()
    user_client.delete_from_org(user_id, org_id)

    typer.secho("User is successfully deleted from organization", fg=typer.colors.BLUE)


def _get_org_user_id_by_name(org_id: UUID, username: str) -> UUID:
    group_client = GroupClient()
    users = group_client.get_users(org_id, username)
    for user in users:
        if user["username"] == username:
            return UUID(user["id"])
    secho_error_and_exit(f"{username} is not a member of this organization.")


def get_current_org() -> Dict[str, Any]:
    """Get the current organization info."""
    user_group_client = UserGroupClient()

    curr_org_id = get_current_group_id()
    if curr_org_id is None:
        secho_error_and_exit("Organization is not identified. Please login again.")

    org = user_group_client.get_group_info()
    if org["id"] == str(curr_org_id):
        return org

    # org context may be wrong
    secho_error_and_exit("Failed to identify organization.")


@app.command()
def members():
    """List up members in the current working organization."""
    group_client = GroupClient()
    org_id = get_current_group_id()

    if org_id is None:
        secho_error_and_exit("Organization is not identified. Please login again.")

    members = group_client.list_users(org_id)
    member_table_formatter.render(members)
