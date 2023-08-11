# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Modules that configures bindings to providers."""

from __future__ import annotations

from injector import Binder, Module

from periflow.utils import url


class URLModule(Module):
    """PeriFlow client module."""

    def configure(self, binder: Binder) -> None:
        """Configures bindings for clients."""
        binder.bind(url.URLProvider, to=url.ProductionURLProvider)  # type: ignore


default_modules = [URLModule]
