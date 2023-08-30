# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Client Dependency Injector."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Optional, Type, Union

from injector import Binder, Injector, Module

_InstallableModuleType = Union[Callable[[Binder], None], Module, Type[Module]]

default_injector = Injector()
test_injector = None


def get_injector() -> Injector:
    """Gets injector for DI."""
    return test_injector or default_injector


@contextmanager
def patch_modules(
    modules: Union[
        _InstallableModuleType, Iterable[_InstallableModuleType], None
    ] = None,
    override_defaults: bool = False,
) -> Iterator[None]:
    """Patches injector modules."""
    global test_injector  # pylint: disable=global-statement
    old_injector = test_injector

    parent_injector: Optional[Injector] = None
    if old_injector:
        parent_injector = old_injector
    elif override_defaults:
        parent_injector = default_injector
    else:
        parent_injector = None

    test_injector = Injector(modules, parent=parent_injector)
    try:
        yield
    finally:
        test_injector = old_injector


def set_default_modules(
    modules: Union[
        _InstallableModuleType, Iterable[_InstallableModuleType], None
    ] = None,
):
    """Set default DI modules."""
    global default_injector  # pylint: disable=global-statement
    default_injector = Injector(modules)
