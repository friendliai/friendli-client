# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Credential CLI."""

# pylint: disable=line-too-long

from __future__ import annotations

import json
import sys
from uuid import UUID

import typer

from periflow.configurator.credential import CredentialInteractiveConfigurator
from periflow.enums import CredType
from periflow.formatter import PanelFormatter, TableFormatter
from periflow.sdk.resource.credential import Credential as CredentialAPI
from periflow.utils.format import secho_error_and_exit

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)
create_app = typer.Typer(
    help="Create credential",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)

app.add_typer(create_app, name="create")

table_formatter = TableFormatter(
    name="Credentials",
    fields=["id", "name", "type", "created_at", "owner_type"],
    headers=["ID", "Name", "Type", "Created At", "Scope"],
)

panel_formatter = PanelFormatter(
    name="Overview",
    fields=["id", "name", "type", "created_at", "owner_type"],
    headers=["ID", "Name", "Type", "Created At", "Scope"],
)


S3_DOC_LINK = (
    "https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html"
)
AZURE_BLOB_DOC_LINK = "https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?tabs=azure-portal"  # pylint: disable=line-too-long
GCP_DOC_LINK = (
    "https://cloud.google.com/iam/docs/creating-managing-service-account-keys"
)


@create_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    interactive: bool = typer.Option(
        False, "--interactive", help="Use interactive mode."
    ),
):
    """Common callback for the interactive option."""
    if not interactive:
        if ctx.invoked_subcommand is None:
            secho_error_and_exit(
                f"You should provide one of the credential types {[ e.value for e in CredType ]}"
                " as an argument, or use '--interactive' option. "
                "Run 'pf credential --help' for more details."
            )
        return

    configurator = CredentialInteractiveConfigurator()

    configurator.start_interaction()
    name, cred_type, value = configurator.render()

    cred = CredentialAPI.create(cred_type=cred_type, name=name, value=value)

    typer.secho("Credential created successfully!", fg=typer.colors.BLUE)
    panel_formatter.render([cred.model_dump()])
    sys.exit(0)


@create_app.command()
def s3(
    name: str = typer.Option(..., "--name", "-n", help="Name of the new credential."),
    aws_access_key_id: str = typer.Option(
        ..., help=f"[AWS] AWS Acess Key ID. Please refer to {S3_DOC_LINK}."
    ),
    aws_secret_access_key: str = typer.Option(
        ..., help=f"[AWS] AWS Secret Access Key. Please refer to {S3_DOC_LINK}."
    ),
    aws_default_region: str = typer.Option(
        ..., help=f"[AWS] Default region name. Please refer to {S3_DOC_LINK}."
    ),
):
    """Creates a credential for AWS S3.

    To use checkpoints in the S3 buckets of your own AWS account, you must provide
    the credential to access the S3 buckets.

    :::tip
    How to get the value of AWS access key ID, AWS secret access key, and AWS default
    region:

    1. Go to the AWS console website.
    2. Sign in with your AWS account with the S3 bucket you want to use.
    3. Go to the [security credential page](https://console.aws.amazon.com/iam/home?#security_credential) and click **Create access key** at the **AWS IAM credentials** tab.
    4. Get a new access key ID and secret access key.
    5. For the AWS default region, use the AWS region code in the [region code table](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-regions).
    :::

    :::info
    You should have a proper IAM permission to read objects in a S3 bucket that you want to use in PeriFlow. Please refer to the [AWS IAM user guide](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_s3_rw-bucket.html) to learn how to set the IAM policy for S3.
    :::

    """
    value = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_default_region": aws_default_region,
    }
    cred = CredentialAPI.create(cred_type=CredType.S3, name=name, value=value)
    panel_formatter.render([cred.model_dump()])


@create_app.command()
def azure_blob(
    name: str = typer.Option(..., "--name", "-n", help="Name of the new credential."),
    storage_account_name: str = typer.Option(
        ..., help="[Azure] Azure Blob storage account name."
    ),
    storage_account_key: str = typer.Option(
        ...,
        help=f"[Azure] Azure Blob storage account access key. Please refer to {AZURE_BLOB_DOC_LINK}.",
    ),
):
    """Creates a credential for Azure Blob storage.

    To use checkpoints in the Azure Blob Storage container, you must provide the
    credential to access the storage account.

    :::tip
    To get the value of storage account key:

    1. Go to Azure portal website.
    2. Sign in your Azure account with the storage you want to use.
    3. Go to the **Storage account** page and click the storage account you want.
    4. At the side navigation tab, click **Access keys**.
    5. Copy the **Key** value for any key between **key1** and **key2**.
    6. Use that value for the `--storage-acount-key` option.
    :::

    :::info
    Azure Blob Storage has three types of resources (in a hierarchical order):

    - The storage account
    - A container in the storage account
    - A blob in a container

    If you create an `azure-blob` credential, you can use the credential for all
    containers in the storage account. Please refer to the [Azure Blob Storage documentation](https://docs.microsoft.com/azure/storage/blobs/storage-blobs-introduction#blob-storage-resources)
    for more information.
    :::

    """
    value = {
        "storage_account_name": storage_account_name,
        "storage_account_key": storage_account_key,
    }
    cred = CredentialAPI.create(cred_type=CredType.BLOB, name=name, value=value)
    panel_formatter.render([cred.model_dump()])


@create_app.command()
def gcs(
    name: str = typer.Option(..., "--name", "-n", help="Name of the new credential."),
    service_account_key_file: typer.FileText = typer.Option(
        ...,
        help=f"[GCP] Path to GCP Service Account Key JSON file. Please refer to {GCP_DOC_LINK}.",
    ),
):
    r"""Creates a credential for Google Cloud Storage.

    To use checkpoints in the Google Cloud Storage, you must create a
    credential to access the GCS buckets.


    :::tip
    How to get a JSON file containing your GCP service account key for the `--service-account-key-file` option:

    1. Go to Google Cloud Platform Console, and sign in.
    2. Go to the **Service accounts** page.
    3. Select the project with the bucket you want to use.
    4. On the **Service accounts** page, click the email address of the service account whose keys you want to list.
    5. Click **Keys**. The console should display a list of keys for your service account, including metadata for each key.
    6. Click the **Add key** dropdown menu and select **Create new key**.
    7. For **Key type**, select "JSON" and click **Create**, which downloads a JSON file with the following format. Use the path to that file for the `--service-account-key-file` option.

    ```json
    {
        "type": "service_account",
        "project_id": "project-id",
        "private_key_id": "key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nprivate-key\n-----END PRIVATE KEY-----\n",
        "client_email": "service-account-email",
        "client_id": "client-id",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://accounts.google.com/o/oauth2/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/service-account-email"
    }
    ```
    :::

    :::info
    You should have a proper IAM role to read objects in a bucket you want to use in PeriFlow. Please see the [GCS IAM role documentation](https://cloud.google.com/storage/docs/access-control/iam-roles).
    :::

    """
    try:
        value = json.load(service_account_key_file)
    except json.JSONDecodeError as exc:
        secho_error_and_exit(f"Error occurred while parsing JSON file... {exc}")
    del value["type"]
    cred = CredentialAPI.create(cred_type=CredType.GCS, name=name, value=value)
    panel_formatter.render([cred.model_dump()])


# pylint: disable=redefined-builtin
@app.command()
def list(
    cred_type: CredType = typer.Option(
        None, "--cred-type", "-t", help="Filter credentials by the type."
    ),
):
    """Lists credentials."""
    creds = CredentialAPI.list(cred_type=cred_type)
    table_formatter.render([cred.model_dump() for cred in creds])


@app.command()
def update(
    credential_id: UUID = typer.Argument(..., help="ID of credential to update.")
):
    """Updates data of a credential.

    Only interactive mode is supported for now.

    """
    configurator = CredentialInteractiveConfigurator()
    configurator.start_interaction_for_update(credential_id)
    name, _, value = configurator.render()

    cred = CredentialAPI.edit(id=credential_id, name=name, value=value)
    typer.secho("Credential updated successfully!", fg=typer.colors.BLUE)
    panel_formatter.render([cred.model_dump()])


@app.command()
def delete(
    credential_id: UUID = typer.Argument(..., help="ID of credential to delete."),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Forcefully deletes credential without confirmation prompt.",
    ),
):
    """Deletes a credential."""
    if not force:
        do_delete = typer.confirm("Are you sure to delete credential?")
        if not do_delete:
            raise typer.Abort()

    CredentialAPI.delete(id=credential_id)

    typer.secho(
        f"Credential ({credential_id}) is deleted successfully!", fg=typer.colors.BLUE
    )
