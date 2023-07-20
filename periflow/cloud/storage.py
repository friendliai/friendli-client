# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Cloud Services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import boto3
from azure.storage.blob import BlobServiceClient
from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client
from typing_extensions import TypeAlias

from periflow.enums import CloudType, StorageType
from periflow.utils.format import secho_error_and_exit

_CloudClient: TypeAlias = Union[S3Client, BlobServiceClient]
T = TypeVar("T", bound=_CloudClient)


storage_type_map: Dict[StorageType, str] = {
    StorageType.S3: "aws",
    StorageType.BLOB: "azure.blob",
    StorageType.GCS: "gcp",
    StorageType.FAI: "fai",
}


storage_type_map_inv: Dict[str, StorageType] = {
    "aws": StorageType.S3,
    "azure.blob": StorageType.BLOB,
    "gcp": StorageType.GCS,
    "fai": StorageType.FAI,
}


GCP_REGION_NAMES = [
    "asia-east1-a",
    "asia-east1-b",
    "asia-east1-c",
    "asia-east2-a",
    "asia-east2-b",
    "asia-east2-c",
    "asia-northeast1-a",
    "asia-northeast1-b",
    "asia-northeast1-c",
    "asia-northeast2-a",
    "asia-northeast2-b",
    "asia-northeast2-c",
    "asia-northeast3-a",
    "asia-northeast3-b",
    "asia-northeast3-c",
    "asia-south1-a",
    "asia-south1-b",
    "asia-south1-c",
    "asia-south2-a",
    "asia-south2-b",
    "asia-south2-c",
    "asia-southeast1-a",
    "asia-southeast1-b",
    "asia-southeast1-c",
    "asia-southeast2-a",
    "asia-southeast2-b",
    "asia-southeast2-c",
    "australia-southeast1-a",
    "australia-southeast1-b",
    "australia-southeast1-c",
    "australia-southeast2-a",
    "australia-southeast2-b",
    "australia-southeast2-c",
    "europe-central2-a",
    "europe-central2-b",
    "europe-central2-c",
    "europe-north1-a",
    "europe-north1-b",
    "europe-north1-c",
    "europe-west1-b",
    "europe-west1-c",
    "europe-west1-d",
    "europe-west2-a",
    "europe-west2-b",
    "europe-west2-c",
    "europe-west3-a",
    "europe-west3-b",
    "europe-west3-c",
    "europe-west4-a",
    "europe-west4-b",
    "europe-west4-c",
    "europe-west6-a",
    "europe-west6-b",
    "europe-west6-c",
    "northamerica-northeast1-a",
    "northamerica-northeast1-b",
    "northamerica-northeast1-c",
    "northamerica-northeast2-a",
    "northamerica-northeast2-b",
    "northamerica-northeast2-c",
    "southamerica-east1-a",
    "southamerica-east1-b",
    "southamerica-east1-c",
    "southamerica-west1-a",
    "southamerica-west1-b",
    "southamerica-west1-c",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
    "us-east1-b",
    "us-east1-c",
    "us-east1-d",
    "us-east4-a",
    "us-east4-b",
    "us-east4-c",
    "us-west1-a",
    "us-west1-b",
    "us-west1-c",
    "us-west2-a",
    "us-west2-b",
    "us-west2-c",
    "us-west3-a",
    "us-west3-b",
    "us-west3-c",
    "us-west4-a",
    "us-west4-b",
]

AZURE_REGION_NAMES = [
    "eastus",
    "eastus2",
    "southcentralus",
    "westus2",
    "westus3",
    "australiaeast",
    "southeastasia",
    "northeurope",
    "swedencentral",
    "uksouth",
    "westeurope",
    "centralus",
    "northcentralus",
    "westus",
    "southafricanorth",
    "centralindia",
    "eastasia",
    "japaneast",
    "jioindiawest",
    "koreacentral",
    "canadacentral",
    "francecentral",
    "germanywestcentral",
    "norwayeast",
    "switzerlandnorth",
    "uaenorth",
    "brazilsouth",
    "centralusstage",
]


AWS_REGION_NAMES = [
    "us-east-2",
    "us-east-1",
    "us-west-1",
    "us-west-2",
    "af-south-1",
    "ap-east-1",
    "ap-south-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-south-1",
    "ap-northeast-3",
    "ap-northeast-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-northeast-1",
    "ca-central-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "eu-south-1",
    "eu-west-3",
    "eu-south-1",
    "eu-north-1",
    "eu-central-2",
    "me-south-1",
    "me-central-1",
    "sa-east-1",
    "us-gov-east-1",
    "us-gov-west-1",
]


FAI_REGION_NAMES = [
    "",
]


storage_region_map = {
    StorageType.S3: AWS_REGION_NAMES,
    StorageType.BLOB: AZURE_REGION_NAMES,
    StorageType.GCS: GCP_REGION_NAMES,
    StorageType.FAI: FAI_REGION_NAMES,
}


cloud_region_map = {
    CloudType.AWS: AWS_REGION_NAMES,
    CloudType.AZURE: AZURE_REGION_NAMES,
    CloudType.GCP: GCP_REGION_NAMES,
}


class CloudStorageClient(ABC, Generic[T]):
    """Cloud storage client interface."""

    def __init__(self, client: T) -> None:
        """Initialize cloud storage client."""
        self.client = client

    @abstractmethod
    def list_storage_files(
        self, storage_name: str, path_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all file objects in the storage.

        Args:
            storage_name (str): Storage name
            path_prefix (Optional[str], optional): Direcotry path under the storage.

        Returns:
            List[Dict[str, Any]]: A list of object info.

        """


class AWSCloudStorageClient(CloudStorageClient[S3Client]):
    """AWS S3 client."""

    def _check_aws_bucket_exists(self, storage_name: str) -> bool:
        try:
            self.client.head_bucket(Bucket=storage_name)
            return True
        except ClientError:
            # include both Forbidden access, Not Exists
            return False

    def list_storage_files(
        self, storage_name: str, path_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List objects in the storage."""
        if not self._check_aws_bucket_exists(storage_name):
            secho_error_and_exit(f"Bucket {storage_name} does not exist")

        file_list = []
        if path_prefix is None:
            resp = self.client.list_objects(Bucket=storage_name)
        else:
            resp = self.client.list_objects(Bucket=storage_name, Prefix=path_prefix)

        if "Contents" not in resp:
            secho_error_and_exit(
                f"No file exists at {path_prefix} in the bucket({storage_name})"
            )
        object_contents = resp["Contents"]
        for object_content in object_contents:
            try:
                object_key = object_content["Key"]
                name = object_key.split("/")[-1]
                if not name:
                    continue  # skip directory
                file_list.append(
                    {
                        "name": name,
                        "path": object_key,
                        "mtime": object_content["LastModified"].isoformat(),
                        "size": object_content["Size"],
                    }
                )
            except KeyError:
                secho_error_and_exit("Unexpected S3 error")

        if not file_list:
            secho_error_and_exit(f"No file exists in Bucket {storage_name}")

        return file_list


class AzureCloudStorageClient(CloudStorageClient[BlobServiceClient]):
    """Azure blob storage client."""

    def list_storage_files(
        self, storage_name: str, path_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List objects in the storage."""
        container_client = self.client.get_container_client(storage_name)
        if not container_client.exists():
            secho_error_and_exit(f"Container {storage_name} does not exist")

        file_list = []
        prefix_option = (
            {"name_starts_with": path_prefix} if path_prefix is not None else {}
        )
        object_contents = container_client.list_blobs(**prefix_option)
        for object_content in object_contents:
            object_name = object_content["name"]
            name = object_name.split("/")[-1]
            if not name:
                continue  # skip directory
            file_list.append(
                {
                    "name": name,
                    "path": object_name,
                    "mtime": object_content["last_modified"].isoformat(),
                    "size": object_content["size"],
                }
            )

        if not file_list:
            secho_error_and_exit(f"No file exists in Bucket {storage_name}")

        return file_list


def build_s3_client(credential_json: Dict[str, str]) -> S3Client:
    """Build AWS S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=credential_json["aws_access_key_id"],
        aws_secret_access_key=credential_json["aws_secret_access_key"],
        region_name=credential_json.get("aws_default_region", None),
    )


def build_blob_client(credential_json: Dict[str, str]) -> BlobServiceClient:
    """Build Azure blob storage client."""
    url = f"https://{credential_json['storage_account_name']}.blob.core.windows.net/"
    return BlobServiceClient(
        account_url=url, credential=credential_json["storage_account_key"]
    )


# TODO: Add GCP support
vendor_client_map: Dict[
    StorageType,
    Tuple[Type[CloudStorageClient], Callable[[Dict[str, str]], _CloudClient]],
] = {
    StorageType.S3: (AWSCloudStorageClient, build_s3_client),
    StorageType.BLOB: (AzureCloudStorageClient, build_blob_client),
}


def build_storage_client(
    vendor: StorageType, credential_json: Dict[str, Any]
) -> CloudStorageClient:
    """Build a cloud storage client."""
    cls, client_build_fn = vendor_client_map[vendor]
    client = client_build_fn(credential_json)
    return cls(client)
