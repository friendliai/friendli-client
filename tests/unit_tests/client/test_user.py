# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test UserClient Service"""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime

import pytest
import requests_mock
import typer

from periflow.client.user import (
    UserAccessKeyClient,
    UserClient,
    UserGroupClient,
    UserMFAClient,
)
from periflow.enums import GroupRole, ProjectRole


@pytest.fixture
def user_client(user_project_group_context) -> UserClient:
    return UserClient()


@pytest.fixture
def user_group_client(user_project_group_context) -> UserGroupClient:
    return UserGroupClient()


@pytest.fixture
def user_mfa() -> UserMFAClient:
    return UserMFAClient()


@pytest.fixture
def user_access_key_client(user_project_group_context) -> UserAccessKeyClient:
    return UserAccessKeyClient()


def test_user_initiate_mfa(
    requests_mock: requests_mock.Mocker, user_mfa: UserMFAClient
):
    # Success
    url_template = deepcopy(user_mfa.url_template)
    url_template.attach_pattern("challenge/$mfa_type")
    requests_mock.post(url_template.render(mfa_type="totp"), status_code=204)
    try:
        user_mfa.initiate_mfa(mfa_type="totp", mfa_token="MFA_TOKEN")
    except typer.Exit:
        raise pytest.fail("Test initiate MFA failed.")

    # Failed
    requests_mock.post(url_template.render(mfa_type="totp"), status_code=400)
    with pytest.raises(typer.Exit):
        user_mfa.initiate_mfa(mfa_type="totp", mfa_token="MFA_TOKEN")


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_client_change_password(
    requests_mock: requests_mock.Mocker, user_client: UserClient
):
    # Success
    url_template = deepcopy(user_client.url_template)
    url_template.attach_pattern(f"{user_client.user_id}/password")
    requests_mock.put(url_template.render(**user_client.url_kwargs), status_code=204)
    try:
        user_client.change_password("1234", "5678")
    except typer.Exit:
        raise pytest.fail("Test change password failed.")

    # Failed
    requests_mock.put(url_template.render(**user_client.url_kwargs), status_code=400)
    with pytest.raises(typer.Exit):
        user_client.change_password("1234", "5678")


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_client_set_group_privilege(
    requests_mock: requests_mock.Mocker, user_client: UserClient
):
    # Success
    url_template = deepcopy(user_client.url_template)
    url_template.attach_pattern("$pf_user_id/pf_group/$pf_group_id/privilege_level")

    user_id = uuid.uuid4()
    group_id = uuid.uuid4()

    requests_mock.patch(
        url_template.render(pf_user_id=user_id, pf_group_id=group_id), status_code=204
    )
    try:
        user_client.set_group_privilege(group_id, user_id, GroupRole.OWNER)
    except typer.Exit:
        raise pytest.fail("Test set group privilege failed.")

    # Failed
    requests_mock.patch(
        url_template.render(pf_user_id=user_id, pf_group_id=group_id), status_code=404
    )
    with pytest.raises(typer.Exit):
        user_client.set_group_privilege(group_id, user_id, GroupRole.OWNER)


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_client_get_project_membership(
    requests_mock: requests_mock.Mocker, user_client: UserClient
):
    # Success
    url_template = deepcopy(user_client.url_template)
    url_template.attach_pattern(f"{user_client.user_id}/pf_project/$pf_project_id")

    project_id = uuid.uuid4()
    user_data = {
        "id": str(user_client.user_id),
        "name": "test",
        "access_level": "admin",
        "created_at": "2022-06-30T06:30:46.896Z",
        "updated_at": "2022-06-30T06:30:46.896Z",
    }

    requests_mock.get(url_template.render(pf_project_id=project_id), json=user_data)

    assert user_client.get_project_membership(project_id) == user_data

    # Failed
    requests_mock.get(url_template.render(pf_project_id=project_id), status_code=404)
    with pytest.raises(typer.Exit):
        user_client.get_project_membership(project_id)


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_client_add_to_project(
    requests_mock: requests_mock.Mocker, user_client: UserClient
):
    # Success
    url_template = deepcopy(user_client.url_template)
    url_template.attach_pattern("$pf_user_id/pf_project/$pf_project_id")

    user_id = uuid.uuid4()
    project_id = uuid.uuid4()

    requests_mock.post(
        url_template.render(pf_user_id=user_id, pf_project_id=project_id),
        status_code=204,
    )
    try:
        user_client.add_to_project(user_id, project_id, ProjectRole.ADMIN)
    except typer.Exit:
        raise pytest.fail("Test add to project failed.")

    # Failed
    requests_mock.post(
        url_template.render(pf_user_id=user_id, pf_project_id=project_id),
        status_code=404,
    )
    with pytest.raises(typer.Exit):
        user_client.add_to_project(user_id, project_id, ProjectRole.ADMIN)


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_client_delete_from_project(
    requests_mock: requests_mock.Mocker, user_client: UserClient
):
    # Success
    url_template = deepcopy(user_client.url_template)
    url_template.attach_pattern("$pf_user_id/pf_project/$pf_project_id")

    user_id = uuid.uuid4()
    project_id = uuid.uuid4()

    requests_mock.delete(
        url_template.render(pf_user_id=user_id, pf_project_id=project_id),
        status_code=204,
    )
    try:
        user_client.delete_from_project(user_id, project_id)
    except typer.Exit:
        raise pytest.fail("Test add to project failed.")

    # Failed
    requests_mock.post(
        url_template.render(pf_user_id=user_id, pf_project_id=project_id),
        status_code=404,
    )
    with pytest.raises(typer.Exit):
        user_client.add_to_project(user_id, project_id, ProjectRole.ADMIN)


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_client_set_project_privilege(
    requests_mock: requests_mock.Mocker, user_client: UserClient
):
    # Success
    url_template = deepcopy(user_client.url_template)
    url_template.attach_pattern("$pf_user_id/pf_project/$pf_project_id/access_level")

    user_id = uuid.uuid4()
    project_id = uuid.uuid4()

    requests_mock.patch(
        url_template.render(pf_user_id=user_id, pf_project_id=project_id),
        status_code=204,
    )
    try:
        user_client.set_project_privilege(user_id, project_id, ProjectRole.ADMIN)
    except typer.Exit:
        raise pytest.fail("Test set project privilege failed.")

    # Failed
    requests_mock.patch(
        url_template.render(pf_user_id=user_id, pf_project_id=project_id),
        status_code=404,
    )
    with pytest.raises(typer.Exit):
        user_client.set_project_privilege(user_id, project_id, ProjectRole.ADMIN)


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_group_client_get_group_info(
    requests_mock: requests_mock.Mocker, user_group_client: UserGroupClient
):
    assert isinstance(user_group_client, UserGroupClient)

    # Success
    url_template = deepcopy(user_group_client.url_template)
    requests_mock.get(
        url_template.render(**user_group_client.url_kwargs),
        json=[{"id": "00000000-0000-0000-0000-000000000000", "name": "my-group"}],
    )
    assert user_group_client.get_group_info() == {
        "id": "00000000-0000-0000-0000-000000000000",
        "name": "my-group",
    }

    # Failed
    requests_mock.get(
        url_template.render(**user_group_client.url_kwargs), status_code=404
    )
    with pytest.raises(typer.Exit):
        user_group_client.get_group_info()


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_access_key_client_create_key(
    requests_mock: requests_mock.Mocker,
    user_access_key_client: UserAccessKeyClient,
):
    assert isinstance(user_access_key_client, UserAccessKeyClient)

    result = {
        "id": "1",
        "name": "test",
        "created_at": str(datetime.now().astimezone()),
        "token": "test-token",
    }

    # Success
    url_template = deepcopy(user_access_key_client.url_template)
    requests_mock.post(
        url_template.render(
            **user_access_key_client.url_kwargs,
            path=f"22222222-2222-2222-2222-222222222222/api_key",
            name="test",
        ),
        json=result,
    )
    assert user_access_key_client.create_access_key("test") == result

    # Failed due to HTTP error
    requests_mock.post(
        url_template.render(
            **user_access_key_client.url_kwargs,
            path=f"22222222-2222-2222-2222-222222222222/api_key",
            name="test",
        ),
        json=result,
        status_code=400,
    )
    with pytest.raises(typer.Exit):
        user_access_key_client.create_access_key("test")


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_access_key_client_list_key(
    requests_mock: requests_mock.Mocker,
    user_access_key_client: UserAccessKeyClient,
):
    assert isinstance(user_access_key_client, UserAccessKeyClient)
    user_id = "22222222-2222-2222-2222-222222222222"
    result = [
        {"id": "1", "name": "test", "created_at": str(datetime.now().astimezone())}
    ]

    # Success
    url_template = deepcopy(user_access_key_client.url_template)
    requests_mock.get(
        url_template.render(
            **user_access_key_client.url_kwargs, path=f"{user_id}/api_key"
        ),
        json=result,
    )
    assert user_access_key_client.list_access_keys() == result

    # Failed due to HTTP error
    url_template = deepcopy(user_access_key_client.url_template)
    requests_mock.get(
        url_template.render(
            **user_access_key_client.url_kwargs, path=f"{user_id}/api_key"
        ),
        json=result,
        status_code=400,
    )
    with pytest.raises(typer.Exit):
        user_access_key_client.list_access_keys()


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_user_access_key_client_delete_key(
    requests_mock: requests_mock.Mocker,
    user_access_key_client: UserAccessKeyClient,
):
    assert isinstance(user_access_key_client, UserAccessKeyClient)
    access_key_id = "33333333-3333-3333-3333-333333333333"

    # Success
    url_template = deepcopy(user_access_key_client.url_template)
    requests_mock.delete(
        url_template.render(
            **user_access_key_client.url_kwargs,
            pk=None,
            path=f"api_key/{access_key_id}",
        )
    )
    user_access_key_client.delete_access_key(access_key_id)

    # Failed due to HTTP error
    requests_mock.delete(
        url_template.render(
            **user_access_key_client.url_kwargs,
            pk=None,
            path=f"api_key/{access_key_id}",
        ),
        status_code=400,
    )
    with pytest.raises(typer.Exit):
        user_access_key_client.delete_access_key(access_key_id)
