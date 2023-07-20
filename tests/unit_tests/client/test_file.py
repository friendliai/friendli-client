# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Unit test for file client."""

from __future__ import annotations

from typing import Iterator
from uuid import uuid4

import pytest
import requests_mock
import typer

from periflow.client.file import FileClient, GroupProjectFileClient


@pytest.fixture
def file_client() -> FileClient:
    return FileClient()


@pytest.fixture
def group_file_client(
    user_project_group_context: Iterator[None],
) -> GroupProjectFileClient:
    return GroupProjectFileClient()


class TestFileClient:
    """Unit test for `FileClientService`."""

    @pytest.mark.usefixtures("patch_auto_token_refresh")
    def test_get_misc_file_upload_url(
        self,
        requests_mock: requests_mock.Mocker,
        file_client: FileClient,
        fake_s3_presigned_url: str,
    ):
        misc_file_id = uuid4()
        resp_body = {
            "upload_url": fake_s3_presigned_url,
        }
        base_url = file_client.url_template.get_base_url()

        requests_mock.post(
            f"{base_url}/files/{misc_file_id}/upload/",
            json=resp_body,
        )
        assert (
            file_client.get_misc_file_upload_url(misc_file_id=misc_file_id)
            == fake_s3_presigned_url
        )

        requests_mock.post(f"{base_url}/files/{misc_file_id}/upload/", status_code=404)
        with pytest.raises(typer.Exit):
            file_client.get_misc_file_upload_url(misc_file_id=misc_file_id)

    @pytest.mark.usefixtures("patch_auto_token_refresh")
    def test_get_misc_file_download_url(
        self,
        requests_mock: requests_mock.Mocker,
        file_client: FileClient,
        fake_s3_presigned_url: str,
    ):
        misc_file_id = uuid4()
        resp_body = {
            "download_url": fake_s3_presigned_url,
        }
        base_url = file_client.url_template.get_base_url()

        requests_mock.post(
            f"{base_url}/files/{misc_file_id}/download/",
            json=resp_body,
        )
        assert (
            file_client.get_misc_file_download_url(misc_file_id=misc_file_id)
            == fake_s3_presigned_url
        )

        requests_mock.post(
            f"{base_url}/files/{misc_file_id}/download/", status_code=404
        )
        with pytest.raises(typer.Exit):
            file_client.get_misc_file_download_url(misc_file_id=misc_file_id)

    @pytest.mark.usefixtures("patch_auto_token_refresh")
    def test_make_misc_file_uploaded(
        self,
        requests_mock: requests_mock.Mocker,
        file_client: FileClient,
    ):
        misc_file_id = uuid4()
        resp_body = {
            "id": str(uuid4()),
            "name": "string",
            "path": "string",
            "mtime": "2023-03-31T06:29:58.703Z",
            "size": 0,
            "uploaded": True,
            "organization_id": str(uuid4()),
            "project_id": str(uuid4()),
            "user_id": str(uuid4()),
        }
        base_url = file_client.url_template.get_base_url()

        requests_mock.patch(
            f"{base_url}/files/{misc_file_id}/uploaded/",
            json=resp_body,
        )
        assert (
            file_client.make_misc_file_uploaded(misc_file_id=misc_file_id) == resp_body
        )

        requests_mock.patch(
            f"{base_url}/files/{misc_file_id}/uploaded/", status_code=404
        )
        with pytest.raises(typer.Exit):
            file_client.make_misc_file_uploaded(misc_file_id=misc_file_id)


class TestGroupProjectFileClient:
    """Unit test for `GroupProjectFileClientService`."""

    @pytest.mark.usefixtures("patch_auto_token_refresh")
    def test_make_create_misc_file(
        self,
        requests_mock: requests_mock.Mocker,
        group_file_client: GroupProjectFileClient,
    ):
        resp_body = {
            "id": str(uuid4()),
            "name": "string",
            "path": "string",
            "mtime": "2023-03-31T06:29:58.703Z",
            "size": 0,
            "uploaded": False,
            "organization_id": str(uuid4()),
            "project_id": str(uuid4()),
            "user_id": str(uuid4()),
        }
        url = group_file_client.url_template.render(**group_file_client.url_kwargs)
        file_info = {}

        requests_mock.post(
            url,
            json=resp_body,
        )
        assert group_file_client.create_misc_file(file_info=file_info) == resp_body

        requests_mock.post(url, status_code=409)
        with pytest.raises(typer.Exit):
            group_file_client.create_misc_file(file_info=file_info)
