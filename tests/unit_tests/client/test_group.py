# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test Group Client."""

from __future__ import annotations

import uuid
from copy import deepcopy

import pytest
import requests_mock

from friendli.client.group import GroupClient
from friendli.errors import APIError


@pytest.fixture
def group_client(user_project_group_context) -> GroupClient:
    return GroupClient()


@pytest.mark.usefixtures("patch_safe_request")
def test_group_client_create_group(
    requests_mock: requests_mock.Mocker, group_client: GroupClient
):
    group_data = {
        "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "pf_group_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "name": "test",
        "created_at": "0001-01-01T00:00:00",
        "updated_at": "0001-01-01T00:00:00",
    }

    # Success
    requests_mock.post(group_client.url_template.render(), json=group_data)
    assert group_client.create_group("test") == group_data

    # Failed due to HTTP error
    requests_mock.post(group_client.url_template.render(), status_code=404)
    with pytest.raises(APIError):
        group_client.create_group("name")


@pytest.mark.usefixtures("patch_safe_request")
def test_group_client_get_group(
    requests_mock: requests_mock.Mocker, group_client: GroupClient
):
    group_id = uuid.UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6")
    group_data = {
        "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "name": "string",
        "status": "staged",
        "hosting_type": "hosted",
        "created_at": "0001-01-01T00:00:00",
        "updated_at": "0001-01-01T00:00:00",
    }

    # Success
    requests_mock.get(group_client.url_template.render(group_id), json=group_data)
    assert group_client.get_group(group_id) == group_data

    # Failed due to HTTP error
    requests_mock.get(group_client.url_template.render(group_id), status_code=404)
    with pytest.raises(APIError):
        group_client.get_group(group_id)


@pytest.mark.usefixtures("patch_safe_request")
def test_group_client_invite_to_group(
    requests_mock: requests_mock.Mocker, group_client: GroupClient
):
    group_id = uuid.UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6")

    # Success
    url_template = deepcopy(group_client.url_template)
    url_template.attach_pattern("$pf_group_id/invite/signup")
    requests_mock.post(url_template.render(pf_group_id=group_id), status_code=204)
    group_client.invite_to_group(group_id, "test@test.com")

    # Failed due to HTTP error
    requests_mock.post(url_template.render(pf_group_id=group_id), status_code=404)
    with pytest.raises(APIError):
        group_client.invite_to_group(group_id, "test@test.com")
