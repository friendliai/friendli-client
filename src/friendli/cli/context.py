# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Application context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from typer import get_app_dir

from .const import APP_NAME
from .typer_util import TyperContext

if TYPE_CHECKING:
    from types import EllipsisType, TracebackType
    from typing import Self


@dataclass
class RootContextObj:
    """Root context for the application."""

    base_url: str | None = None
    token: str | None = None


class TyperAppContext(TyperContext[RootContextObj]):
    """Typer's application context."""


class AppContext:
    """Application context."""

    def __init__(self, root_ctx: RootContextObj) -> None:
        """Initialize application context."""
        self.root = root_ctx

        from rich.console import Console
        from rich.theme import Theme

        from ..sdk.sync import SyncClient
        from .backend.auth import PatAuthBackend
        from .backend.settings import SettingsBackend

        app_dir = get_app_dir(APP_NAME)
        self.settings_backend = SettingsBackend(Path(app_dir))
        self.auth_backend = PatAuthBackend()

        # get auth
        auth = None
        if user_info := self.settings_backend.settings.user_info:
            auth = self.auth_backend.fetch_credential(user_info.user_id)

        auth = auth or root_ctx.token
        self.sdk = SyncClient(auth=auth, base_url=root_ctx.base_url)
        theme = Theme(
            {
                "info": "dim cyan",
                "warn": "magenta",
                "danger": "bold red",
                "success": "bold green",
                "headline": "bold bright_blue",
                "subheadline": "dim bright_blue",
                "content": "bright_white",
                "highlight": "bold bright_yellow",
            }
        )
        self.console = Console(theme=theme)

    def refresh_client(self, auth: str | EllipsisType | None = ...) -> None:
        """Refresh sync client."""
        if auth is not ...:
            self.root.token = auth

        self.sdk.refresh_http_client(auth=self.root.token, base_url=self.root.base_url)

    def __enter__(self) -> Self:
        """Context manager for application context."""
        self.sdk.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.sdk.__exit__(exc_type, exc_val, exc_tb)
