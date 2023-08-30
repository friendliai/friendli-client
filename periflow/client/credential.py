# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CredentialClient Service."""


from __future__ import annotations

from string import Template
from typing import Any, Dict, Optional
from uuid import UUID

from periflow.client.base import Client, safe_request
from periflow.enums import CredType
from periflow.utils.maps import cred_type_map


class CredentialClient(Client[UUID]):
    """Credential client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_auth_uri("credential"))

    def get_credential(self, credential_id: UUID) -> Dict[str, Any]:
        """Get a credential info."""
        response = safe_request(self.retrieve, err_prefix="Credential is not found.")(
            pk=credential_id
        )
        return response.json()

    def update_credential(
        self,
        credential_id: UUID,
        *,
        name: Optional[str] = None,
        type_version: Optional[str] = None,
        value: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a credential."""
        request_data: Dict[str, Any] = {}
        if name is not None:
            request_data["name"] = name
        if type_version is not None:
            request_data["type_version"] = type_version
        if value is not None:
            request_data["value"] = value
        response = safe_request(
            self.partial_update, err_prefix="Failed to updated credential"
        )(pk=credential_id, json=request_data)
        return response.json()

    def delete_credential(self, credential_id: UUID) -> None:
        """Delete a credential."""
        safe_request(self.delete, err_prefix="Failed to delete credential")(
            pk=credential_id
        )


class CredentialTypeClient(Client):
    """Credential type client."""

    @property
    def url_path(self) -> Template:
        """Get an URL path."""
        return Template(self.url_provider.get_training_uri("credential_type/"))

    def get_schema_by_type(self, cred_type: CredType) -> Optional[Dict[str, Any]]:
        """Get a credential JSON schema."""
        type_name = cred_type_map[cred_type]
        response = safe_request(
            self.list, err_prefix="Failed to get credential schema."
        )()
        for cred_type_json in response.json():
            if cred_type_json["type_name"] == type_name:
                return cred_type_json["versions"][-1][
                    "schema"
                ]  # use the latest version
        return None
