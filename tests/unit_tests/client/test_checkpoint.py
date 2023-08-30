# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test CheckpointClient Service"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
import requests_mock
import typer

from periflow.client.checkpoint import CheckpointClient, CheckpointFormClient
from periflow.utils.fs import S3_MPU_PART_MAX_SIZE, S3_UPLOAD_SIZE_LIMIT


@pytest.fixture
def checkpoint_client() -> CheckpointClient:
    return CheckpointClient()


@pytest.fixture
def checkpoint_form_client() -> CheckpointFormClient:
    return CheckpointFormClient()


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_checkpoint_client_get_checkpoint(
    requests_mock: requests_mock.Mocker, checkpoint_client: CheckpointClient
):
    assert isinstance(checkpoint_client, CheckpointClient)
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
    with pytest.raises(typer.Exit):
        assert checkpoint_client.get_checkpoint(checkpoint_id)


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_checkpoint_client_delete_checkpoint(
    requests_mock: requests_mock.Mocker, checkpoint_client: CheckpointClient
):
    assert isinstance(checkpoint_client, CheckpointClient)
    checkpoint_id = uuid4()

    url_template = deepcopy(checkpoint_client.url_template)
    url_template.attach_pattern("$checkpoint_id/")

    # Success
    requests_mock.delete(
        url_template.render(checkpoint_id=checkpoint_id), status_code=204
    )
    try:
        checkpoint_client.delete_checkpoint(checkpoint_id)
    except typer.Exit:
        raise pytest.fail("Checkpoint delete test failed.")

    # Failed due to HTTP error
    requests_mock.delete(
        url_template.render(checkpoint_id=checkpoint_id), status_code=404
    )
    with pytest.raises(typer.Exit):
        checkpoint_client.delete_checkpoint(checkpoint_id)


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_checkpoint_client_restore_checkpoint(
    requests_mock: requests_mock.Mocker, checkpoint_client: CheckpointClient
):
    assert isinstance(checkpoint_client, CheckpointClient)
    checkpoint_id = uuid4()

    url_template = deepcopy(checkpoint_client.url_template)
    url_template.attach_pattern("$checkpoint_id/restore/")

    # Success
    requests_mock.post(
        url_template.render(checkpoint_id=checkpoint_id), status_code=204, json={}
    )
    try:
        checkpoint_client.restore_checkpoint(checkpoint_id)
    except typer.Exit:
        raise pytest.fail("Checkpoint restore test failed.")

    # Failed due to HTTP error
    requests_mock.post(
        url_template.render(checkpoint_id=checkpoint_id), status_code=404, json={}
    )
    with pytest.raises(typer.Exit):
        assert checkpoint_client.restore_checkpoint(checkpoint_id)


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_checkpoint_client_get_checkpoint_download_urls(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    assert isinstance(checkpoint_form_client, CheckpointFormClient)
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
    with pytest.raises(typer.Exit):
        assert checkpoint_form_client.get_checkpoint_download_urls(ckpt_form_id)


@pytest.mark.usefixtures("patch_auto_token_refresh")
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
    with pytest.raises(typer.Exit):
        checkpoint_form_client.update_checkpoint_files(
            ckpt_form_id=ckpt_form_id, files=files
        )


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_checkpoint_client_upload(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    assert isinstance(checkpoint_form_client, CheckpointFormClient)
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
        checkpoint_form_client.get_spu_urls(
            obj_id=ckpt_form_id,
            storage_paths=paths,
        )
        == resp_body
    )
    requests_mock.post(url, status_code=404)
    with pytest.raises(typer.Exit):
        checkpoint_form_client.get_spu_urls(
            obj_id=ckpt_form_id,
            storage_paths=paths,
        )


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_checkpoint_client_start_multipart_upload(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    assert isinstance(checkpoint_form_client, CheckpointFormClient)
    ckpt_form_id = uuid4()
    base_url = checkpoint_form_client.url_template.get_base_url()
    url = f"{base_url}/model_forms/{ckpt_form_id}/start_mpu/"
    paths = ["pytorch_model.bin"]
    fake_upload_id = "fakeuploadid"
    file_size = math.ceil(S3_UPLOAD_SIZE_LIMIT * 1.7)
    resp_body = {
        "path": paths[0],
        "upload_id": fake_upload_id,
        "upload_urls": [
            {
                "upload_url": f"https://s3.download.amazone.com/{paths[0]}-part{part_num}",
                "part_number": part_num,
            }
            for part_num in range(math.ceil(file_size / S3_MPU_PART_MAX_SIZE))
        ],
    }

    with TemporaryDirectory() as dir:
        with open(os.path.join(dir, paths[0]), "wb") as f:
            f.seek(file_size - 1)
            f.write(b"\0")

        requests_mock.post(url, json=resp_body)
        assert checkpoint_form_client.get_mpu_urls(
            obj_id=ckpt_form_id,
            local_paths=[os.path.join(dir, path) for path in paths],
            storage_paths=paths,
        ) == [resp_body]
        requests_mock.post(url, status_code=404)
        with pytest.raises(typer.Exit):
            checkpoint_form_client.get_mpu_urls(
                obj_id=ckpt_form_id,
                local_paths=[os.path.join(dir, path) for path in paths],
                storage_paths=paths,
            )


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_checkpoint_client_complete_multipart_upload(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    assert isinstance(checkpoint_form_client, CheckpointFormClient)
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
    checkpoint_form_client.complete_mpu(
        obj_id=ckpt_form_id, path=path, upload_id=fake_upload_id, parts=parts
    )

    requests_mock.post(url, status_code=404)
    with pytest.raises(typer.Exit):
        checkpoint_form_client.complete_mpu(
            obj_id=ckpt_form_id, path=path, upload_id=fake_upload_id, parts=parts
        )


@pytest.mark.usefixtures("patch_auto_token_refresh")
def test_checkpoint_client_abort_multipart_upload(
    requests_mock: requests_mock.Mocker,
    checkpoint_form_client: CheckpointFormClient,
):
    assert isinstance(checkpoint_form_client, CheckpointFormClient)
    ckpt_form_id = uuid4()
    base_url = checkpoint_form_client.url_template.get_base_url()
    url = f"{base_url}/model_forms/{ckpt_form_id}/abort_mpu/"
    path = "pytorch_model.bin"
    fake_upload_id = "fakeuploadid"

    requests_mock.post(url, status_code=204)
    checkpoint_form_client.abort_mpu(
        obj_id=ckpt_form_id, path=path, upload_id=fake_upload_id
    )

    requests_mock.post(url, status_code=404)
    with pytest.raises(typer.Exit):
        checkpoint_form_client.abort_mpu(
            obj_id=ckpt_form_id, path=path, upload_id=fake_upload_id
        )


def test_actual_s3_upload(
    requests_mock: requests_mock.Mocker, checkpoint_form_client: CheckpointClient
):
    assert isinstance(checkpoint_form_client, CheckpointFormClient)
    ckpt_form_id = uuid4()

    small_file_1_name = "small_1.bin"
    small_file_2_name = "small_2.bin"
    large_file_1_name = "large_1.bin"
    large_file_2_name = "large_2.bin"
    small_file_1_size = S3_UPLOAD_SIZE_LIMIT // 2
    small_file_2_size = S3_UPLOAD_SIZE_LIMIT // 4
    large_file_1_size = math.ceil(S3_UPLOAD_SIZE_LIMIT * 1.1)
    large_file_2_size = math.ceil(S3_UPLOAD_SIZE_LIMIT * 1.5)

    # Mock S3 upload with presigned URLs.
    requests_mock.put(f"https://s3.amazon.com/{small_file_1_name}")
    requests_mock.put(f"https://s3.amazon.com/{small_file_2_name}")
    # Mock S3 multipart upload with presigned URLs.
    # HACK: 1 is added because the actual file read by ``CustomCallbackIOWrapper`` does not occur since the request is mocked.
    large_file_1_total_parts = math.ceil(large_file_1_size / S3_MPU_PART_MAX_SIZE) + 1
    large_file_2_total_parts = math.ceil(large_file_2_size / S3_MPU_PART_MAX_SIZE) + 1
    for part_num in range(large_file_1_total_parts):
        requests_mock.put(
            f"https://s3.amazon.com/{large_file_1_name}/part{part_num}",
            headers={"Etag": f"{large_file_1_name}etag-part{part_num}"},
        )
    for part_num in range(large_file_2_total_parts):
        requests_mock.put(
            f"https://s3.amazon.com/{large_file_2_name}/part{part_num}",
            headers={"Etag": f"{large_file_2_name}etag-part{part_num}"},
        )
    # Mock registry API calls
    base_url = checkpoint_form_client.url_template.get_base_url()
    complete_mpu_url = f"{base_url}/model_forms/{ckpt_form_id}/complete_mpu/"
    requests_mock.post(complete_mpu_url, status_code=204)
    abort_mpu_url = f"{base_url}/model_forms/{ckpt_form_id}/abort_mpu/"
    requests_mock.post(abort_mpu_url, status_code=204)

    with TemporaryDirectory() as dir:
        with open(os.path.join(dir, small_file_1_name), "wb") as f_1, open(
            os.path.join(dir, small_file_2_name), "wb"
        ) as f_2, open(os.path.join(dir, large_file_1_name), "wb") as f_3, open(
            os.path.join(dir, large_file_2_name), "wb"
        ) as f_4:
            f_1.seek(small_file_1_size - 1)
            f_1.write(b"\0")
            f_2.seek(small_file_2_size - 1)
            f_2.write(b"\0")
            f_3.seek(large_file_1_size - 1)
            f_3.write(b"\0")
            f_4.seek(large_file_2_size - 1)
            f_4.write(b"\0")

        checkpoint_form_client.upload_files(
            obj_id=ckpt_form_id,
            spu_url_dicts=[
                {
                    "path": small_file_1_name,
                    "upload_url": f"https://s3.amazon.com/{small_file_1_name}",
                },
                {
                    "path": small_file_2_name,
                    "upload_url": f"https://s3.amazon.com/{small_file_2_name}",
                },
            ],
            mpu_url_dicts=[
                {
                    "path": large_file_1_name,
                    "upload_id": f"fakeuploadid{large_file_1_name}",
                    "upload_urls": [
                        {
                            "upload_url": f"https://s3.amazon.com/{large_file_1_name}/part{part_num}",
                            "part_number": part_num,
                        }
                        for part_num in range(large_file_1_total_parts)
                    ],
                },
                {
                    "path": large_file_2_name,
                    "upload_id": f"fakeuploadid{large_file_2_name}",
                    "upload_urls": [
                        {
                            "upload_url": f"https://s3.amazon.com/{large_file_2_name}/part{part_num}",
                            "part_number": part_num,
                        }
                        for part_num in range(large_file_2_total_parts)
                    ],
                },
            ],
            source_path=Path(dir),
        )
