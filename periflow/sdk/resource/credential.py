# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Credential SDK."""

# pylint: disable=line-too-long, arguments-differ, too-many-arguments, redefined-builtin

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import ValidationError

from periflow.client.credential import CredentialClient
from periflow.client.project import ProjectCredentialClient
from periflow.enums import CredType
from periflow.errors import InvalidConfigError, NotSupportedError
from periflow.schema.resource.v1.credential import V1Credential
from periflow.sdk.resource.base import ResourceAPI
from periflow.utils.maps import cred_schema_map, cred_type_map_inv


class Credential(ResourceAPI[V1Credential, UUID]):
    """PeriFlow credential resource API."""

    @staticmethod
    def create(cred_type: CredType, name: str, value: Dict[str, Any]) -> V1Credential:
        """Creates a new credential.

        Args:
            cred_type (CredType): The type of credential.
            name (str): The name of credential to create.
            value (Dict[str, Any]): Values of credential in plaintext.

        Raises:
            InvalidConfigError: Raised if `value` does not match the credential schema.

        Returns:
            V1Credential: The created credential object.

        """
        cred_schema = cred_schema_map[cred_type]
        try:
            cred_schema.model_validate(obj=value)
        except ValidationError as exc:
            raise InvalidConfigError(
                detail=f"Wrong credential format: {str(exc)}"
            ) from exc

        client = ProjectCredentialClient()
        raw_cred = client.create_credential(cred_type, name, 1, value)
        raw_cred["type"] = cred_type_map_inv[raw_cred["type"]].value
        cred = V1Credential.model_validate(raw_cred)
        return cred

    @staticmethod
    def get(id: UUID, *args, **kwargs) -> V1Credential:
        """[skip-doc] Gets a credential info.

        Args:
            id (UUID): ID of credential to retrieve.

        Returns:
            Dict[str, Any]: Retrieved credential info.

        """
        msg = "Getting a specific credential info is not supported."
        raise NotSupportedError(msg)

    @staticmethod
    def list(cred_type: Optional[CredType] = None) -> List[V1Credential]:
        """Lists credentials.

        Args:
            cred_type (Optional[CredType], optional): Filters by credential types. Defaults to None.

        Returns:
            List[V1Credential]: A list of retrieved credentials.

        """
        client = ProjectCredentialClient()
        raw_creds = client.list_credentials(cred_type)
        creds = []
        for raw_cred in raw_creds:
            cred = V1Credential.model_validate(raw_cred)
            cred.type = cred_type_map_inv[cred.type].value
            creds.append(cred)
        return creds

    @staticmethod
    def edit(
        id: UUID,
        name: Optional[str] = None,
        value: Optional[Dict[str, Any]] = None,
    ) -> V1Credential:
        """Edits credential info.

        Args:
            id (UUID): ID of credential to edit.
            name (Optional[str], optional): Name to update. No update if it is `None`. Defaults to None.
            value (Optional[Dict[str, Any]], optional): The new value of credential in plaintext. No update if it is `None`. Defaults to None.

        Returns:
            V1Credential: Updated credential object.

        """
        client = CredentialClient()
        raw_cred = client.update_credential(
            id, name=name, type_version="1", value=value
        )
        cred = V1Credential.model_validate(raw_cred)
        cred.type = cred_type_map_inv[cred.type].value
        return cred

    @staticmethod
    def delete(id: UUID) -> None:
        """Deletes a credential.

        Args:
            id (UUID): ID of credential to delete.

        """
        client = CredentialClient()
        client.delete_credential(id)
