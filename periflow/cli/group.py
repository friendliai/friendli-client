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
    fields=["id", "name", "email", "privilege_level"],
    headers=["ID", "Name", "Email", "Role"],
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


@app.command("set-role")
def set_role(
    email: str = typer.Argument(..., help="Email of the user to assign a role"),
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

    user_id = get_org_user_id_by_email(org["id"], email)
    user_client.set_group_privilege(org["id"], user_id, role)
    typer.echo(
        f"Organization role for user '{email}' successfully updated to {role.value}!"
    )


@app.command("delete-user")
def delete_user(
    email: str = typer.Argument(
        ...,
        help="Email of the user to delete from the organization",
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
    user_id = get_org_user_id_by_email(org_id, email)

    if not force:
        do_delete = typer.confirm(
            f"Are you sure to remove user '{email}' from the organization?"
        )
        if not do_delete:
            raise typer.Abort()
    user_client.delete_from_org(user_id, org_id)

    typer.secho("User is successfully deleted from organization", fg=typer.colors.BLUE)


def get_org_user_id_by_email(org_id: UUID, email: str) -> UUID:
    """Get ID of user by the email."""
    group_client = GroupClient()
    users = group_client.list_users(org_id)
    for user in users:
        if user["email"] == email:
            return UUID(user["id"])
    secho_error_and_exit(f"User '{email}' is not a member of this organization.")


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
