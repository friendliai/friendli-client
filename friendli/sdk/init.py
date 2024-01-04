# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Initialize Friendli."""

# pylint: disable=line-too-long

from __future__ import annotations

import friendli
from friendli.client.project import ProjectClient, find_project_id
from friendli.client.user import UserGroupClient, UserGroupProjectClient
from friendli.errors import AuthorizationError


def init(token: str, project: str) -> None:
    """Setup Friendli authorization info.

    All Friendli APIs followed by this function will be executed with the API key,
    provided in the `token` argument, and project context provided in the `project`
    argument. If you call the Friendli APIs without calling this function, the context
    set by `friendli login` CLI command will be used.

    Args:
        token (str): Friendli API key.
        project (str): Project name.

    Raises:
        AuthorizationError: Raised if the API key, provided in the `token` argument, does not have permission to any organization or to the project provided in the `project` argument.

    """
    friendli.token = token

    user_group_client = UserGroupClient()

    try:
        org = user_group_client.get_group_info()
    except IndexError as exc:
        raise AuthorizationError(
            "Does have not permission to any organization."
        ) from exc
    friendli.team_id = org["id"]

    user_group_project_client = UserGroupProjectClient()
    project_client = ProjectClient()

    project_id = find_project_id(
        projects=user_group_project_client.list_projects(),
        project_name=project,
    )
    if project_client.check_project_membership(pf_project_id=project_id):
        friendli.project_id = str(project_id)
    else:
        raise AuthorizationError(
            f"Does not have permission to the project '{project}'."
        )
