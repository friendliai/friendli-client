# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Project CLI."""

from __future__ import annotations

from typing import Optional, Tuple
from uuid import UUID

import typer

from periflow.cli.group import get_current_org, get_org_user_id_by_email
from periflow.client.group import GroupProjectClient
from periflow.client.project import ProjectClient, find_project_id
from periflow.client.user import UserClient, UserGroupProjectClient
from periflow.context import (
    get_current_project_id,
    project_context_path,
    set_current_project_id,
)
from periflow.enums import GroupRole, ProjectRole
from periflow.formatter import PanelFormatter, TableFormatter
from periflow.utils.format import secho_error_and_exit

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)
project_table_formatter = TableFormatter(
    name="Project", fields=["name", "id"], headers=["Name", "ID"]
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
def list(
    tail: Optional[int] = typer.Option(
        None, "--tail", help="The number of project list to view at the tail"
    ),
    head: Optional[int] = typer.Option(
        None, "--tail", help="The number of project list to view at the head"
    ),
    show_group_project: bool = typer.Option(
        False, "--group", "-g", help="Show all projects in the current group"
    ),
):
    """List all accessible projects."""
    if show_group_project:
        client = GroupProjectClient()
    else:
        client = UserGroupProjectClient()  # type: ignore

    projects = client.list_projects()
    current_project_id = get_current_project_id()

    for project in projects:
        if current_project_id is not None and project["id"] == str(current_project_id):
            project["name"] = f"[bold green]* {project['name']}"
            project["id"] = f"[bold green]{project['id']}"
        else:
            project["name"] = f"  {project['name']}"

    if tail is not None or head is not None:
        target_project_list = []
        if tail is not None:
            target_project_list.extend(projects[:tail])
        if head is not None:
            target_project_list.extend(
                projects[-head:]  # pylint: disable=invalid-unary-operand-type
            )
    else:
        target_project_list = projects

    project_table_formatter.render(target_project_list)


@app.command()
def create(name: str = typer.Argument(..., help="Name of project to create")):
    """Create a new project.

    :::info
    Every resource within a project, such as credentials, checkpoints, and deployments,
    is shared with project members.
    :::

    """
    client = GroupProjectClient()
    project_detail = client.create_project(name)
    project_panel_formatter.render(project_detail)


@app.command()
def current():
    """Get the current working project."""
    client = ProjectClient()
    project_id = get_current_project_id()
    if project_id is None:
        secho_error_and_exit("working project is not set")
    project = client.get_project(project_id)
    project_panel_formatter.render(project)


@app.command()
def switch(
    name: str = typer.Argument(
        ...,
        help="Name of project to switch",
    )
):
    """Switch working project context.

    There can exist multiple projects under the organization, and each organization
    member can be included in multiple projects. You first need to select the working
    project context where you want to use the services.

    :::info
    Your working project context is written at `./periflow/project` file in your home directory.
    :::

    """
    user_group_project_client = UserGroupProjectClient()
    project_client = ProjectClient()

    project_id = find_project_id(user_group_project_client.list_projects(), name)
    if project_client.check_project_membership(pf_project_id=project_id):
        set_current_project_id(project_id)
        typer.secho(f"Project switched to {name}.", fg=typer.colors.BLUE)
    else:
        secho_error_and_exit(
            f"You don't have permission to project ({name}). Please contact to the project admin."
        )


@app.command()
def delete(
    name: str = typer.Argument(
        ...,
        help="Name of project to delete",
    )
):
    """Delete the project."""
    project_client = ProjectClient()
    user_group_project_client = UserGroupProjectClient()
    project_id = find_project_id(user_group_project_client.list_projects(), name)
    project_client.delete_project(pf_project_id=project_id)
    if project_id == get_current_project_id():
        project_context_path.unlink()
    typer.secho(f"Project {name} deleted.", fg=typer.colors.BLUE)


def _check_project_and_get_id() -> Tuple[UUID, UUID]:
    """Get org_id and project_id if valid."""
    user_client = UserClient()

    org = get_current_org()
    project_id = get_current_project_id()
    if project_id is None:
        secho_error_and_exit("Failed to identify project... Please set project again.")

    if org["privilege_level"] == GroupRole.OWNER:
        return UUID(org["id"]), project_id

    requester = user_client.get_project_membership(project_id)
    if requester["access_level"] != ProjectRole.ADMIN:
        secho_error_and_exit("Only the admin of the project can add-user/set-role")

    return org["id"], project_id


@app.command("add-user")
def add_user(
    email: str = typer.Argument(
        ...,
        help="Email of the user to add to the current working project",
    ),
    role: ProjectRole = typer.Argument(
        ...,
        help="Project role to assign",
    ),
):
    """Add user to project.

    If you are the **Admin** of a project, you can add members in the same organization
    to the project and assign their roles in the project.

    :::info
    If you have **Owner** role in the organization, you have full-access to all projects
    inside the organization.
    :::

    """
    user_client = UserClient()

    org_id, project_id = _check_project_and_get_id()
    user_id = get_org_user_id_by_email(org_id, email)

    user_client.add_to_project(user_id, project_id, role)
    typer.secho(
        f"User '{email}' is successfully added to project", fg=typer.colors.BLUE
    )


@app.command("delete-user")
def delete_user(
    email: str = typer.Argument(
        ...,
        help="Email of the user to delete from the current working project",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Forcefully delete without confirmation prompt",
    ),
):
    """Delete user from the project.

    :::info
    Only organization **Owner** or project **Admin** can evict members from the project.
    :::

    """
    user_client = UserClient()

    org_id, project_id = _check_project_and_get_id()
    user_id = get_org_user_id_by_email(org_id, email)

    if not force:
        do_delete = typer.confirm(
            f"Are you sure to remove user '{email}' from the project?"
        )
        if not do_delete:
            raise typer.Abort()

    user_client.delete_from_project(user_id, project_id)
    typer.secho(
        f"User '{email}' is successfully deleted from project", fg=typer.colors.BLUE
    )


@app.command("set-role")
def set_role(
    email: str = typer.Argument(
        ...,
        help="Email of the user to assign a project role",
    ),
    role: ProjectRole = typer.Argument(
        ...,
        help="Project role",
    ),
):
    """Set project role for the user.

    Refer to the following table for the project-level roles and privileges.

    | | Admin | Maintainer | Developer | Guest |
    |-|-|-|-|-|
    | Add organization members to the project | ✓ | ✗ | ✗ | ✗ |
    | Assign project roles | ✓ | ✗ | ✗ | ✗ |
    | Delete project | ✓ | ✗ | ✗ | ✗ |
    | Delete & edit checkpoints | ✓ | ✗ | ✗ | ✗ |
    | Delete & edit credentials, deployments | ✓ | ✓ | ✗ | ✗ |
    | Creaete credentials | ✓ | ✓ | ✗ | ✗ |
    | Create checkpoints, deployments | ✓ | ✓ | ✓ | ✗ |
    | Read checkpoints, credentials, deployments | ✓ | ✓ | ✓ | ✓ |
    | Send inference requests to deployments | ✓ | ✓ | ✓ | ✓ |

    :::info
    Only the **Admin** users can assign roles to project members:
    :::

    :::info
    If you have the **Owner** role in the organization, you have full-access to all
    projects inside the organization.
    :::

    """
    user_client = UserClient()

    org_id, project_id = _check_project_and_get_id()
    user_id = get_org_user_id_by_email(org_id, email)

    user_client.set_project_privilege(user_id, project_id, role)
    typer.secho(
        f"Project role for user '{email}' successfully updated to {role.value}!"
    )


@app.command()
def members():
    """List project members."""
    project_client = ProjectClient()

    project_id = get_current_project_id()
    if project_id is None:
        secho_error_and_exit("Failed to identify project... Please set project again.")

    members = project_client.list_users(project_id)
    member_table_formatter.render(members)
