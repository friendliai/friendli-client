# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""CLI application settings."""

from __future__ import annotations

import keyring

from ...const import APP_NAME


class PatAuthBackend:
    """Authentication backend."""

    # TODO: implement plain text fallback
    # https://github.com/jaraco/keyrings.alt/blob/main/keyrings/alt/file.py

    def __init__(self) -> None:
        """Initialize."""
        # FIXME(AJ): handle when keyring does not exist
        self.kb = keyring.get_keyring()

    def store_credential(self, user_id: str, token: str) -> None:
        """Store personal access token in keychain."""
        self.kb.set_password(APP_NAME, user_id, token)

    def fetch_credential(self, user_id: str) -> str | None:
        """Fetch personal access token from keychain."""
        return self.kb.get_password(APP_NAME, user_id)

    def clear_credential(self, user_id: str) -> None:
        """Clear personal access token from keychain."""
        self.kb.delete_password(APP_NAME, user_id)
