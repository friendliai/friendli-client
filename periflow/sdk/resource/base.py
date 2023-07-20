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

    @staticmethod
    @abstractmethod
    def create(*args, **kwargs) -> R:
        """Creates a resource."""

    @staticmethod
    @abstractmethod
    def get(id: RI, *args, **kwargs) -> R:
        """Gets a specific resource."""

    @staticmethod
    @abstractmethod
    def list(*args, **kwargs) -> List[R]:
        """Lists reousrces."""
