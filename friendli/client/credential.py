# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Credential Clients."""


from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from friendli.client.base import Client
from friendli.enums import CredType
from friendli.utils.maps import cred_type_map


class CredentialClient(Client[UUID]):
    """Credential client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_auth_uri("credential")

    def get_credential(self, credential_id: UUID) -> Dict[str, Any]:
        """Get a credential info."""
        data = self.retrieve(
            pk=credential_id,
        )
        return data

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
        data = self.partial_update(
            pk=credential_id,
            json=request_data,
        )
        return data

    def delete_credential(self, credential_id: UUID) -> None:
        """Delete a credential."""
        self.delete(
            pk=credential_id,
        )


class CredentialTypeClient(Client):
    """Credential type client."""

    @property
    def url_path(self) -> str:
        """Get an URL path."""
        return self.url_provider.get_training_uri("credential_type/")

    def get_schema_by_type(self, cred_type: CredType) -> Optional[Dict[str, Any]]:
        """Get a credential JSON schema."""
        type_name = cred_type_map[cred_type]
        data = self.list(
            pagination=False,
        )
        for cred_type_json in data:
            if cred_type_json["type_name"] == type_name:
                return cred_type_json["versions"][-1][
                    "schema"
                ]  # use the latest version
        return None
