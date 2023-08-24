# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test Checkpoint Client."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict
from uuid import UUID, uuid4

import pytest
import requests_mock

from periflow.client.checkpoint import (
    CheckpointClient,
    CheckpointFormClient,
    GroupProjectCheckpointClient,
)
from periflow.enums import CheckpointCategory, StorageType
from periflow.errors import APIError
from periflow.utils.transfer import S3_MAX_PART_SIZE, S3_MULTIPART_THRESHOLD


@pytest.fixture
def checkpoint_client() -> CheckpointClient:
    return CheckpointClient()


@pytest.fixture
def checkpoint_form_client() -> CheckpointFormClient:
    return CheckpointFormClient()


@pytest.fixture
def group_project_checkpoint_client(
    user_project_group_context,
) -> GroupProjectCheckpointClient:
    return GroupProjectCheckpointClient()


@pytest.mark.usefixtures("patch_safe_request")
def test_checkpoint_client_get_checkpoint(
    requests_mock: requests_mock.Mocker, checkpoint_client: CheckpointClient
):
    checkpoint_id = uuid4()
    ckpt_form_id = uuid4()
    resp_body = {
        "id": str(checkpoint_id),
        "forms": [
            {
                "id": str(ckpt_form_id),
            },
        ],
    }

    url_template = deepcopy(checkpoint_client.url_template)
    url_template.attach_pattern("$checkpoint_id/")

    # Success
    requests_mock.get(
        url_template.render(checkpoint_id=checkpoint_id),
        json=resp_body,
    )
    assert checkpoint_client.get_checkpoint(checkpoint_id) == resp_body
    assert checkpoint_client.get_first_checkpoint_form(checkpoint_id) == ckpt_form_id

    # Failed due to HTTP error
    requests_mock.get(url_template.render(checkpoint_id=checkpoint_id), status_code=404)
    with pytest.raises(APIError):
        assert checkpoint_client.get_checkpoint(checkpoint_id)


@pytest.mark.usefixtures("patch_safe_request")
def test_checkpoint_client_delete_checkpoint(
    requests_mock: requests_mock.Mocker, checkpoint_client: CheckpointClient
):
    checkpoint_id = uuid4()

    url_template = deepcopy(checkpoint_client.url_template)
    url_template.attach_pattern("$checkpoint_id/")

    # Success
    requests_mock.delete(
        url_template.render(checkpoint_id=checkpoint_id), status_code=204
    )
    checkpoint_client.delete_checkpoint(checkpoint_id)

    # Failed due to HTTP error
    requests_mock.delete(
        url_template.render(checkpoint_id=checkpoint_id), status_code=404
    )
    with pytest.raises(APIError):
        checkpoint_client.delete_checkpoint(checkpoint_id)


@pytest.mark.usefixtures("patch_safe_request")
def test_checkpoint_client_restore_checkpoint(
    requests_mock: requests_mock.Mocker, checkpoint_client: CheckpointClient
):
    checkpoint_id = uuid4()

    url_template = deepcopy(checkpoint_client.url_template)
    url_template.attach_pattern("$checkpoint_id/restore/")

    # Success
    requests_mock.post(
        url_template.render(checkpoint_id=checkpoint_id), status_code=204, json={}
    )
    checkpoint_client.restore_checkpoint(checkpoint_id)

    # Failed due to HTTP error
    requests_mock.post(
        url_template.render(checkpoint_id=checkpoint_id), status_code=404, json={}
    )
    with pytest.raises(APIError):
        assert checkpoint_client.restore_checkpoint(checkpoint_id)


@pytest.mark.usefixtures("patch_safe_request")
def test_checkpoint_client_get_checkpoint_download_urls(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    ckpt_form_id = uuid4()
    base_url = checkpoint_form_client.url_template.get_base_url()

    data = {
        "files": [
            {
                "name": "new_ckpt_1000.pth",
                "path": "ckpt/new_ckpt_1000.pth",
                "mtime": "2022-04-20T06:27:37.907Z",
                "size": 2048,
                "download_url": "https://s3.download.url.com",
            }
        ]
    }

    # Success
    requests_mock.get(
        f"{base_url}/model_forms/{ckpt_form_id}/download/",
        json=data,
    )
    assert (
        checkpoint_form_client.get_checkpoint_download_urls(ckpt_form_id)
        == data["files"]
    )

    # Failed due to HTTP error
    requests_mock.get(
        f"{base_url}/model_forms/{ckpt_form_id}/download/", status_code=404
    )
    with pytest.raises(APIError):
        assert checkpoint_form_client.get_checkpoint_download_urls(ckpt_form_id)


@pytest.mark.usefixtures("patch_safe_request")
def test_checkpoint_client_update_checkpoint_files(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    ckpt_form_id = uuid4()
    base_url = checkpoint_form_client.url_template.get_base_url()
    files = [
        {
            "name": "new_ckpt_1000.pth",
            "path": "ckpt/new_ckpt_1000.pth",
            "mtime": "2022-04-20T06:27:37.907Z",
            "size": 2048,
            "download_url": "https://s3.download.url.com",
        }
    ]
    resp_body = {"files": files}

    requests_mock.patch(f"{base_url}/model_forms/{ckpt_form_id}/", json=resp_body)
    assert (
        checkpoint_form_client.update_checkpoint_files(
            ckpt_form_id=ckpt_form_id, files=files
        )
        == resp_body
    )

    requests_mock.patch(f"{base_url}/model_forms/{ckpt_form_id}/", status_code=404)
    with pytest.raises(APIError):
        checkpoint_form_client.update_checkpoint_files(
            ckpt_form_id=ckpt_form_id, files=files
        )


@pytest.mark.usefixtures("patch_safe_request")
def test_checkpoint_client_upload(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    ckpt_form_id = uuid4()
    base_url = checkpoint_form_client.url_template.get_base_url()
    url = f"{base_url}/model_forms/{ckpt_form_id}/upload/"
    paths = ["model.config", "pytorch_model.bin"]
    resp_body = [
        {"path": path, "upload_url": f"https://s3.download.amazone.com/{path}"}
        for path in paths
    ]

    requests_mock.post(url, json=resp_body)
    assert (
        checkpoint_form_client.get_upload_urls(
            obj_id=ckpt_form_id,
            storage_paths=paths,
        )
        == resp_body
    )
    requests_mock.post(url, status_code=404)
    with pytest.raises(APIError):
        checkpoint_form_client.get_upload_urls(
            obj_id=ckpt_form_id,
            storage_paths=paths,
        )


@pytest.mark.usefixtures("patch_safe_request")
def test_checkpoint_client_start_multipart_upload(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    ckpt_form_id = uuid4()
    base_url = checkpoint_form_client.url_template.get_base_url()
    url = f"{base_url}/model_forms/{ckpt_form_id}/start_mpu/"
    paths = ["pytorch_model.bin"]
    fake_upload_id = "fakeuploadid"
    file_size = math.ceil(S3_MULTIPART_THRESHOLD * 1.7)
    resp_body = {
        "path": paths[0],
        "upload_id": fake_upload_id,
        "upload_urls": [
            {
                "upload_url": f"https://s3.download.amazone.com/{paths[0]}-part{part_num}",
                "part_number": part_num,
            }
            for part_num in range(math.ceil(file_size / S3_MAX_PART_SIZE))
        ],
    }

    with TemporaryDirectory() as dir:
        with open(os.path.join(dir, paths[0]), "wb") as f:
            f.seek(file_size - 1)
            f.write(b"\0")

        requests_mock.post(url, json=resp_body)
        assert checkpoint_form_client.get_multipart_upload_urls(
            obj_id=ckpt_form_id,
            local_paths=[os.path.join(dir, path) for path in paths],
            storage_paths=paths,
        ) == [resp_body]
        requests_mock.post(url, status_code=404)
        with pytest.raises(APIError):
            checkpoint_form_client.get_multipart_upload_urls(
                obj_id=ckpt_form_id,
                local_paths=[os.path.join(dir, path) for path in paths],
                storage_paths=paths,
            )


@pytest.mark.usefixtures("patch_safe_request")
def test_checkpoint_client_complete_multipart_upload(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    ckpt_form_id = uuid4()
    base_url = checkpoint_form_client.url_template.get_base_url()
    url = f"{base_url}/model_forms/{ckpt_form_id}/complete_mpu/"
    path = "pytorch_model.bin"
    fake_upload_id = "fakeuploadid"
    parts = [
        {
            "etag": "fakeetag0",
            "part_number": 0,
        },
        {
            "etag": "fakeetag1",
            "part_number": 1,
        },
    ]

    requests_mock.post(url, status_code=204)
    checkpoint_form_client.complete_multipart_upload(
        obj_id=ckpt_form_id, path=path, upload_id=fake_upload_id, parts=parts
    )

    requests_mock.post(url, status_code=404)
    with pytest.raises(APIError):
        checkpoint_form_client.complete_multipart_upload(
            obj_id=ckpt_form_id, path=path, upload_id=fake_upload_id, parts=parts
        )


@pytest.mark.usefixtures("patch_safe_request")
def test_checkpoint_client_abort_multipart_upload(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    ckpt_form_id = uuid4()
    base_url = checkpoint_form_client.url_template.get_base_url()
    url = f"{base_url}/model_forms/{ckpt_form_id}/abort_mpu/"
    path = "pytorch_model.bin"
    fake_upload_id = "fakeuploadid"

    requests_mock.post(url, status_code=204)
    checkpoint_form_client.abort_multipart_upload(
        obj_id=ckpt_form_id, path=path, upload_id=fake_upload_id
    )

    requests_mock.post(url, status_code=404)
    with pytest.raises(APIError):
        checkpoint_form_client.abort_multipart_upload(
            obj_id=ckpt_form_id, path=path, upload_id=fake_upload_id
        )


@pytest.mark.usefixtures("patch_safe_request")
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


@pytest.mark.usefixtures("patch_safe_request")
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
            credential_id=UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"),
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
            credential_id=UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"),
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
