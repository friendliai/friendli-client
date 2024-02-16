# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Modules that configures bindings to providers."""

from __future__ import annotations

from injector import Binder, Module

from friendli import settings
from friendli.utils import url


class SettingsModule(Module):
    """Settings module."""

    def configure(self, binder: Binder) -> None:
        """Configures bindings for settings."""
        binder.bind(url.URLProvider, to=url.ProductionURLProvider)  # type: ignore
        binder.bind(settings.Settings, to=settings.ProductionSettings)  # type: ignore


default_modules = [SettingsModule]
