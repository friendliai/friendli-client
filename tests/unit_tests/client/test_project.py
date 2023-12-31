# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test Project Client."""

from __future__ import annotations

import pytest
import requests_mock

from friendli.client.project import ProjectCredentialClient
from friendli.enums import CredType
from friendli.errors import APIError


@pytest.fixture
def project_credential_client(
    user_project_group_context,
) -> ProjectCredentialClient:
    return ProjectCredentialClient()


@pytest.mark.usefixtures("patch_safe_request")
def test_project_credential_client_service(
    requests_mock: requests_mock.Mocker,
    project_credential_client: ProjectCredentialClient,
):
    # Sucess
    requests_mock.get(
        project_credential_client.url_template.render(
            **project_credential_client.url_kwargs
        ),
        json=[{"id": 0, "name": "our-s3-secret", "type": "s3"}],
    )
    assert project_credential_client.list_credentials(CredType.S3) == [
        {"id": 0, "name": "our-s3-secret", "type": "s3"}
    ]

    # Failed due to HTTP error
    requests_mock.get(
        project_credential_client.url_template.render(
            **project_credential_client.url_kwargs
        ),
        status_code=400,
    )
    with pytest.raises(APIError):
        project_credential_client.list_credentials(CredType.S3)
