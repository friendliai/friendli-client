# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# Copyright (C) 2021 FriendliAI

"""Test Client Service"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from unittest.mock import Mock

import pytest
import typer
from azure.storage.blob import BlobServiceClient, ContainerClient
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from friendli.cloud.storage import (
    AWSCloudStorageClient,
    AzureCloudStorageClient,
    build_storage_client,
)
from friendli.enums import StorageType


@pytest.fixture
def s3_credential_json() -> Dict[str, Any]:
    return {
        "aws_access_key_id": "fake_aws_access_key_id",
        "aws_secret_access_key": "fake_aws_secret_access_key",
        "aws_default_region": "us-east-1",
    }


@pytest.fixture
def blob_credential_json() -> Dict[str, Any]:
    return {
        "storage_account_name": "fakestorageaccountname",
        "storage_account_key": "fake_storage_account_key",
    }


@pytest.fixture
def blob_client_mock():
    return Mock(BlobServiceClient)()


@pytest.fixture
def container_client():
    return Mock(ContainerClient)()


@pytest.fixture
def s3_client_mock():
    return Mock(BaseClient)()


@pytest.fixture
def aws_storage_helper(s3_client_mock) -> AWSCloudStorageClient:
    return AWSCloudStorageClient(s3_client_mock)


@pytest.fixture
def azure_storage_helper(blob_client_mock) -> AzureCloudStorageClient:
    return AzureCloudStorageClient(blob_client_mock)


def test_build_storage_helper(
    s3_credential_json: Dict[str, Any], blob_credential_json: Dict[str, Any]
):
    aws_storage_helper = build_storage_client(StorageType.S3, s3_credential_json)
    assert isinstance(aws_storage_helper, AWSCloudStorageClient)
    assert isinstance(aws_storage_helper.client, BaseClient)

    azure_storage_helper = build_storage_client(StorageType.BLOB, blob_credential_json)
    assert isinstance(azure_storage_helper, AzureCloudStorageClient)
    assert isinstance(azure_storage_helper.client, BlobServiceClient)


def test_aws_list_storage_files(
    aws_storage_helper: AWSCloudStorageClient, s3_client_mock
):
    # Success
    file_data = [
        {
            "Key": "dir/",
            "LastModified": datetime.utcnow(),
            "ETAG": "e32a59b7-3cc2-4666-bf99-27e238b7cf9c",
            "Size": 0,
            "StorageClass": "STANDARD",
            "Owner": {"ID": "e32a59b7-3cc2-4666-bf99-27e238b7cf9c"},
        },
        {
            "Key": "dir/file_1.txt",
            "LastModified": datetime.utcnow(),
            "ETAG": "e32a59b7-3cc2-4666-bf99-27e238b7cf9c",
            "Size": 2048,
            "StorageClass": "STANDARD",
            "Owner": {"ID": "e32a59b7-3cc2-4666-bf99-27e238b7cf9c"},
        },
        {
            "Key": "file_2.txt",
            "LastModified": datetime.utcnow(),
            "ETAG": "e32a59b7-3cc2-4666-bf99-27e238b7cf9c",
            "Size": 2048,
            "StorageClass": "STANDARD",
            "Owner": {"ID": "e32a59b7-3cc2-4666-bf99-27e238b7cf9c"},
        },
    ]
    s3_client_mock.list_objects.return_value = {"Contents": file_data}

    actual = aws_storage_helper.list_storage_files("my-bucket", "dir")
    expected = [
        {
            "name": d["Key"].split("/")[-1],
            "path": d["Key"],
            "mtime": d["LastModified"].isoformat(),
            "size": d["Size"],
        }
        for d in file_data[1:]
    ]
    assert actual == expected
    s3_client_mock.head_bucket.assert_called_once_with(Bucket="my-bucket")
    s3_client_mock.list_objects.assert_called_once_with(
        Bucket="my-bucket", Prefix="dir"
    )


def test_aws_list_storage_files_bucket_not_exist(
    aws_storage_helper: AWSCloudStorageClient, s3_client_mock
):
    s3_client_mock.head_bucket.side_effect = ClientError(
        {"Error": {"Code": "fake err", "Message": "fake err"}}, "head_bucket"
    )

    with pytest.raises(typer.Exit):
        aws_storage_helper.list_storage_files("my-bucket")
    s3_client_mock.head_bucket.assert_called_once_with(Bucket="my-bucket")


def test_aws_list_storage_files_bucket_contains_no_file(
    aws_storage_helper: AWSCloudStorageClient, s3_client_mock
):
    s3_client_mock.list_objects.return_value = {"Contents": []}
    with pytest.raises(typer.Exit):
        aws_storage_helper.list_storage_files("my-bucket")
    s3_client_mock.head_bucket.assert_called_once_with(Bucket="my-bucket")
    s3_client_mock.list_objects.assert_called_once_with(Bucket="my-bucket")


def test_azure_list_storage_files(
    azure_storage_helper: AzureCloudStorageClient, blob_client_mock, container_client
):
    file_data = [
        {
            "name": "dir/",
            "container": "my-container",
            "last_modified": datetime.utcnow(),
            "size": 0,
        },
        {
            "name": "dir/file_1.txt",
            "container": "my-container",
            "last_modified": datetime.utcnow(),
            "size": 2048,
        },
        {
            "name": "file_2.txt",
            "container": "my-container",
            "last_modified": datetime.utcnow(),
            "size": 2048,
        },
    ]

    blob_client_mock.get_container_client.return_value = container_client
    container_client.exists.return_value = True
    container_client.list_blobs.return_value = file_data

    assert azure_storage_helper.list_storage_files("my-container", "dir") == [
        {
            "name": d["name"].split("/")[-1],
            "path": d["name"],
            "mtime": d["last_modified"].isoformat(),
            "size": d["size"],
        }
        for d in file_data[1:]
    ]
    blob_client_mock.get_container_client.assert_called_once_with("my-container")
    container_client.exists.assert_called_once()
    container_client.list_blobs.assert_called_once_with(name_starts_with="dir")


def test_azure_list_storage_files_container_not_exists(
    azure_storage_helper: AzureCloudStorageClient, blob_client_mock, container_client
):
    blob_client_mock.get_container_client.return_value = container_client
    container_client.exists.return_value = False

    with pytest.raises(typer.Exit):
        azure_storage_helper.list_storage_files("my-container")
    blob_client_mock.get_container_client.assert_called_once_with("my-container")
    container_client.exists.assert_called_once()


def test_azure_list_storage_files_container_no_file(
    azure_storage_helper: AzureCloudStorageClient, blob_client_mock, container_client
):
    blob_client_mock.get_container_client.return_value = container_client
    container_client.exists.return_value = True
    container_client.list_blobs.return_value = []

    with pytest.raises(typer.Exit):
        azure_storage_helper.list_storage_files("my-container")
    blob_client_mock.get_container_client.assert_called_once_with("my-container")
    container_client.exists.assert_called_once()
    container_client.list_blobs.assert_called_once()
