# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Initialize PeriFlow."""

# pylint: disable=line-too-long

from __future__ import annotations

import periflow
from periflow.client.project import ProjectClient, find_project_id
from periflow.client.user import UserGroupClient, UserGroupProjectClient
from periflow.errors import AuthorizationError


def init(api_key: str, project: str) -> None:
    """Setup PeriFlow authorization info.

    All PeriFlow APIs followed by this function will be executed with the API key,
    provided in the `api_key` argument, and project context provided in the `project`
    argument. If you call the PeriFlow APIs without calling this function, the context
    set by `pf login` CLI command will be used.

    Args:
        api_key (str): PeriFlow API key.
        project (str): Project name.

    Raises:
        AuthorizationError: Raised if the API key, provided in the `api_key` argument, does not have permission to any organization or to the project provided in the `project` argument.

    """
    periflow.api_key = api_key

    user_group_client = UserGroupClient()

    try:
        org = user_group_client.get_group_info()
    except IndexError as exc:
        raise AuthorizationError(
            "Does have not permission to any organization."
        ) from exc
    periflow.org_id = org["id"]

    user_group_project_client = UserGroupProjectClient()
    project_client = ProjectClient()

    project_id = find_project_id(
        projects=user_group_project_client.list_projects(),
        project_name=project,
    )
    if project_client.check_project_membership(pf_project_id=project_id):
        periflow.project_id = str(project_id)
    else:
        raise AuthorizationError(
            f"Does not have permission to the project '{project}'."
        )
