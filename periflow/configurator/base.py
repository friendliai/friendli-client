# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Configurator."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Tuple, TypeVar, Union

from jsonschema import Draft7Validator, ValidationError
from typing_extensions import TypeAlias

from periflow.errors import InvalidConfigError

T = TypeVar("T", bound=Union[str, Tuple[Any, ...]])


class InteractiveConfigurator(ABC, Generic[T]):
    """Interative mode interface for configuration."""

    @abstractmethod
    def start_interaction(self) -> None:
        """Start configuration interative prompt."""

    @abstractmethod
    def render(self) -> T:
        """Render the complete configuration."""


IO: TypeAlias = Union[io.TextIOWrapper, io.FileIO, io.BytesIO]


class Configurator(ABC):
    """Mixin class defining the validation interface of configuration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize configurator."""
        self._config = config

    @property
    @abstractmethod
    def validation_schema(self) -> dict:
        """Get a JSON schema for validation."""

    @classmethod
    @abstractmethod
    def from_file(cls, f: IO) -> Configurator:
        """Create a new object from the configuration file."""

    def validate(self) -> None:
        """Validate the configuration."""
        try:
            Draft7Validator(self.validation_schema).validate(self._config)
        except ValidationError as exc:
            raise InvalidConfigError(repr(exc.message)) from exc
