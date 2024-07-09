# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""CLI application settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import ApplicationConfig, UserInfo, WorkingContext

if TYPE_CHECKING:
    from pathlib import Path


class SettingsBackend:
    """Settings backend for the CLI application."""

    def __init__(self, settings_path: Path) -> None:
        """Initialize settings backend."""
        self._settings_path = settings_path
        if settings_path.exists():
            self._settings = ApplicationConfig.model_validate_json(
                settings_path.read_text("utf-8")
            )
        else:
            self._settings = ApplicationConfig(user_info=None)

    @property
    def settings(self) -> ApplicationConfig:
        """Get application settings."""
        return self._settings.copy()

    def save(self, settings: ApplicationConfig) -> None:
        """Save application settings."""
        self._settings = settings
        self._settings_path.write_text(settings.model_dump_json(indent=2), "utf-8")

    def logout(self) -> None:
        """Handle logout."""
        self._settings = self._settings.model_copy(update={"user_info": None})
        self.save(self._settings)
