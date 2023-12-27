# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Interface for resource management SDK."""

# pylint: disable=redefined-builtin

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

R = TypeVar("R")
RI = TypeVar("RI")


class ResourceAPI(ABC, Generic[R, RI]):
    """Abstract class for resource APIs."""

    @abstractmethod
    def create(self, *args, **kwargs) -> R:
        """Creates a resource."""

    @abstractmethod
    def get(self, id: RI, *args, **kwargs) -> R:
        """Gets a specific resource."""

    @abstractmethod
    def list(self, *args, **kwargs) -> List[R]:
        """Lists reousrces."""
