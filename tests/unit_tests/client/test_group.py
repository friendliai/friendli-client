# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test Group Client."""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict

import pytest
import requests_mock

from periflow.client.group import GroupClient, GroupProjectCheckpointClient
from periflow.enums import CheckpointCategory, StorageType
from periflow.errors import APIError


@pytest.fixture
def group_client(user_project_group_context) -> GroupClient:
    return GroupClient()


@pytest.fixture
def group_project_checkpoint_client(
    user_project_group_context,
) -> GroupProjectCheckpointClient:
    return GroupProjectCheckpointClient()


@pytest.mark.usefixtures("patch_auto_token_refresh")
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


@pytest.mark.usefixtures("patch_auto_token_refresh")
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


@pytest.mark.usefixtures("patch_auto_token_refresh")
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


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_group_checkpoint_list_checkpoints(
    requests_mock: requests_mock.Mocker,
    group_project_checkpoint_client: GroupProjectCheckpointClient,
):
    def build_response_item(category: str, vendor: str, region: str) -> Dict[str, Any]:
        return {
            "id": "22222222-2222-2222-2222-222222222222",
            "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "ownerships": [
                {
                    "organization_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "project_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                }
            ],
            "model_category": category,
            "job_id": 2147483647,
            "name": "string",
            "attributes": {
                "job_setting_json": {},
                "data_json": {},
            },
            "forms": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "form_category": "ORCA",
                    "credential_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "vendor": vendor,
                    "region": region,
                    "storage_name": "STORAGE_NAME",
                    "dist_json": {},
                    "files": [
                        {
                            "name": "NAME",
                            "path": "PATH",
                            "mtime": "2022-04-19T09:03:47.352Z",
                            "size": 9,
                        }
                    ],
                }
            ],
            "iteration": 922,
            "created_at": "2022-04-19T09:03:47.352Z",
        }

    job_data = {
        "results": [
            build_response_item("JOB", "aws", "us-east-2"),
        ],
        "next_cursor": None,
    }

    user_data = {
        "results": [
            build_response_item("USER", "aws", "us-east-1"),
        ],
        "next_cursor": None,
    }

    # Success
    url = group_project_checkpoint_client.url_template.render(
        **group_project_checkpoint_client.url_kwargs
    )
    requests_mock.get(url, json=user_data)
    assert (
        group_project_checkpoint_client.list_checkpoints(
            CheckpointCategory.USER_PROVIDED, 10, deleted=False
        )
        == user_data["results"]
    )
    requests_mock.get(url, json=job_data)
    assert (
        group_project_checkpoint_client.list_checkpoints(
            CheckpointCategory.JOB_GENERATED, 10, deleted=False
        )
        == job_data["results"]
    )

    # Failed due to HTTP error
    requests_mock.get(url, status_code=400)
    with pytest.raises(APIError):
        group_project_checkpoint_client.list_checkpoints(
            CheckpointCategory.USER_PROVIDED, 10, deleted=False
        )


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_group_checkpoint_create_checkpoints(
    requests_mock: requests_mock.Mocker,
    group_project_checkpoint_client: GroupProjectCheckpointClient,
):
    data = {
        "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "category": "user_provided",
        "vendor": "aws",
        "region": "us-east-1",
        "credential_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "storage_name": "my-ckpt",
        "iteration": 1000,
        "files": [
            {
                "name": "new_ckpt_1000.pth",
                "path": "ckpt/new_ckpt_1000.pth",
                "mtime": "2022-04-20T06:27:37.907Z",
                "size": 2048,
            }
        ],
    }

    # Success
    # TODO: change after PFA integration
    url = group_project_checkpoint_client.url_template.render(
        **group_project_checkpoint_client.url_kwargs
    )
    requests_mock.post(url, json=data)
    assert (
        group_project_checkpoint_client.create_checkpoint(
            name="my-ckpt",
            vendor=StorageType.S3,
            region="us-east-1",
            credential_id=uuid.UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"),
            iteration=1000,
            storage_name="my-ckpt",
            files=[
                {
                    "name": "new_ckpt_1000.pth",
                    "path": "ckpt/new_ckpt_1000.pth",
                    "mtime": "2022-04-20T06:27:37.907Z",
                    "size": 2048,
                }
            ],
            dist_config={"k": "v"},
            attributes={
                "job_setting_json": {"k": "v"},
                "data_json": {"k": "v"},
            },
        )
        == data
    )
    assert requests_mock.request_history[-1].json() == {
        "job_id": None,
        "vendor": "s3",
        "region": "us-east-1",
        "storage_name": "my-ckpt",
        "model_category": "USER",
        "form_category": "ORCA",
        "name": "my-ckpt",
        "dist_json": {"k": "v"},
        "attributes": {
            "job_setting_json": {"k": "v"},
            "data_json": {"k": "v"},
        },
        "secret_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "secret_type": "credential",
        "iteration": 1000,
        "user_id": "22222222-2222-2222-2222-222222222222",  # TODO: change after PFA integration
        "files": [
            {
                "name": "new_ckpt_1000.pth",
                "path": "ckpt/new_ckpt_1000.pth",
                "mtime": "2022-04-20T06:27:37.907Z",
                "size": 2048,
            }
        ],
    }

    # Failed due to HTTP error
    requests_mock.post(url, status_code=400)
    with pytest.raises(APIError):
        group_project_checkpoint_client.create_checkpoint(
            name="my-ckpt",
            vendor=StorageType.S3,
            region="us-east-1",
            credential_id=uuid.UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"),
            iteration=1000,
            storage_name="my-ckpt",
            files=[
                {
                    "name": "new_ckpt_1000.pth",
                    "path": "ckpt/new_ckpt_1000.pth",
                    "mtime": "2022-04-20T06:27:37.907Z",
                    "size": 2048,
                }
            ],
            dist_config={"k": "v"},
            attributes={"k": "v"},
        )
