# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test Credential Client."""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest
import requests_mock

from periflow.client.credential import CredentialClient, CredentialTypeClient
from periflow.enums import CredType
from periflow.errors import APIError


@pytest.fixture
def credential_client() -> CredentialClient:
    return CredentialClient()


@pytest.fixture
def credential_type_client() -> CredentialTypeClient:
    return CredentialTypeClient()


@pytest.mark.usefixtures("patch_safe_request")
def test_credential_client_get_credential(
    requests_mock: requests_mock.Mocker, credential_client: CredentialClient
):
    cred_id = uuid4()

    # Success
    url_template = deepcopy(credential_client.url_template)
    url_template.attach_pattern("$credential_id")
    requests_mock.get(
        url_template.render(credential_id=cred_id),
        json={"id": 0, "name": "my-s3-secret", "type": "s3"},
    )
    assert credential_client.get_credential(cred_id) == {
        "id": 0,
        "name": "my-s3-secret",
        "type": "s3",
    }

    # Failed due to HTTP error
    requests_mock.get(url_template.render(credential_id=cred_id), status_code=404)
    with pytest.raises(APIError):
        credential_client.get_credential(cred_id)


@pytest.mark.usefixtures("patch_safe_request")
def test_credential_client_update_credential(
    requests_mock: requests_mock.Mocker, credential_client: CredentialClient
):
    cred_id = uuid4()

    # Success
    url_template = deepcopy(credential_client.url_template)
    url_template.attach_pattern("$credential_id")
    requests_mock.patch(
        url_template.render(credential_id=cred_id),
        json={"id": 0, "name": "my-az-blob-secret", "type": "azure.blob"},
    )
    assert credential_client.update_credential(
        cred_id, name="my-az-blob-secret", type_version="1", value={"k": "v"}
    ) == {"id": 0, "name": "my-az-blob-secret", "type": "azure.blob"}
    assert credential_client.update_credential(cred_id) == {  # no updated field
        "id": 0,
        "name": "my-az-blob-secret",
        "type": "azure.blob",
    }

    # Failed due to HTTP error
    requests_mock.patch(url_template.render(credential_id=cred_id), status_code=404)
    with pytest.raises(APIError):
        credential_client.update_credential(
            cred_id, name="my-az-blob-secret", type_version="1", value={"k": "v"}
        )


@pytest.mark.usefixtures("patch_safe_request")
def test_credential_client_delete_credential(
    requests_mock: requests_mock.Mocker, credential_client: CredentialClient
):
    cred_id = uuid4()

    # Success
    url_template = deepcopy(credential_client.url_template)
    url_template.attach_pattern("$credential_id")
    requests_mock.delete(url_template.render(credential_id=cred_id), status_code=204)
    credential_client.delete_credential(cred_id)

    # Failed due to HTTP error
    requests_mock.delete(url_template.render(credential_id=cred_id), status_code=404)
    with pytest.raises(APIError):
        credential_client.delete_credential(cred_id)


@pytest.mark.usefixtures("patch_safe_request")
def test_credential_type_client_get_schema_by_type(
    requests_mock: requests_mock.Mocker,
    credential_type_client: CredentialTypeClient,
):
    assert isinstance(credential_type_client, CredentialTypeClient)

    data = [
        {
            "type_name": "aws",
            "versions": [
                {
                    "type_version": 1,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "aws_access_key_id": {"type": "string", "minLength": 1},
                            "aws_secret_access_key": {"type": "string", "minLength": 1},
                            "aws_default_region": {
                                "type": "string",
                                "examples": [
                                    "us-east-1",
                                    "us-east-2",
                                    "us-west-1",
                                    "us-west-2",
                                    "eu-west-1",
                                    "eu-central-1",
                                    "ap-northeast-1",
                                    "ap-northeast-2",
                                    "ap-southeast-1",
                                    "ap-southeast-2",
                                    "ap-south-1",
                                    "sa-east-1",
                                ],
                            },
                        },
                        "required": [
                            "aws_access_key_id",
                            "aws_secret_access_key",
                            "aws_default_region",
                        ],
                    },
                }
            ],
        },
        {
            "type_name": "gcp",
            "versions": [
                {
                    "type_version": 1,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "default": "service_account"},
                            "project_id": {"type": "string", "minLength": 1},
                            "private_key_id": {"type": "string", "minLength": 1},
                            "private_key": {"type": "string", "minLength": 1},
                            "client_email": {"type": "string", "minLength": 1},
                            "client_id": {"type": "string", "minLength": 1},
                            "auth_uri": {"type": "string", "minLength": 1},
                            "token_uri": {"type": "string", "minLength": 1},
                            "auth_provider_x509_cert_url": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "client_x509_cert_url": {"type": "string", "minLength": 1},
                        },
                        "required": [
                            "project_id",
                            "private_key_id",
                            "private_key",
                            "client_email",
                            "client_id",
                            "auth_uri",
                            "token_uri",
                            "auth_provider_x509_cert_url",
                            "client_x509_cert_url",
                        ],
                    },
                }
            ],
        },
        {
            "type_name": "azure.blob",
            "versions": [
                {
                    "type_version": 1,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "storage_account_name": {
                                "type": "string",
                                "minLength": 3,
                                "maxLength": 24,
                            },
                            "storage_account_key": {"type": "string", "minLength": 1},
                        },
                        "required": ["storage_account_name", "storage_account_key"],
                    },
                }
            ],
        },
    ]

    # Success
    requests_mock.get(credential_type_client.url_template.render(), json=data)
    assert (
        credential_type_client.get_schema_by_type(CredType.S3)
        == data[0]["versions"][-1]["schema"]
    )
    assert (
        credential_type_client.get_schema_by_type(CredType.GCS)
        == data[1]["versions"][-1]["schema"]
    )
    assert (
        credential_type_client.get_schema_by_type(CredType.BLOB)
        == data[2]["versions"][-1]["schema"]
    )

    # Failed due to HTTP error
    requests_mock.get(credential_type_client.url_template.render(), status_code=404)
    with pytest.raises(APIError):
        credential_type_client.get_schema_by_type(CredType.S3)
