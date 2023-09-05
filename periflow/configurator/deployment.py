# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Deployment Configurator."""

from __future__ import annotations

import json
from typing import Any, Dict

from periflow.configurator.base import IO, Configurator
from periflow.utils.format import secho_error_and_exit


class DRCConfigurator(Configurator):
    """Configurator for default request config."""

    @property
    def validation_schema(self) -> dict:
        """JSON schema to validate DRC file contents."""
        return {
            "type": "object",
            "properties": {
                "stop": {"type": "array", "items": {"type": "string"}},
                "stop_tokens": {
                    "type": "object",
                    "properties": {
                        "properties": {
                            "tokens": {"type": "array", "items": {"type": "integer"}}
                        },
                        "required": ["tokens"],
                    },
                },
                "bad_words": {"type": "array", "items": {"type": "string"}},
                "bad_word_tokens": {
                    "type": "object",
                    "properties": {
                        "properties": {
                            "tokens": {"type": "array", "items": {"type": "integer"}}
                        },
                        "required": ["tokens"],
                    },
                },
            },
            "allOf": [
                {"not": {"required": ["stop", "stop_tokens"]}},
                {"not": {"required": ["bad_words", "bad_word_tokens"]}},
            ],
            "minProperties": 1,
            "additionalProperties": False,
        }

    @classmethod
    def from_file(cls, f: IO) -> Configurator:
        """Build a DRC configurator from a DRC file."""
        try:
            config: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            secho_error_and_exit(f"Error occurred while parsing config file: {e!r}")

        return cls(config)  # type: ignore


class DeploymentConfigurator(Configurator):
    """Deployment configurator."""

    @classmethod
    def from_file(cls, f: IO) -> Configurator:
        """Build a DRC configurator from a DRC file."""
        try:
            config: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            secho_error_and_exit(f"Error occurred while parsing config file: {e!r}")

        return cls(config)  # type: ignore


class OrcaDeploymentConfigurator(DeploymentConfigurator):
    """Orca deployment config."""

    @property
    def validation_schema(self) -> dict:
        """JSON schema to validate config."""
        return {
            "type": "object",
            "properties": {
                "max_batch_size": {"type": "integer"},
                "max_token_count": {"type": "integer"},
                "max_num_tokens_to_replace": {"type": "integer"},
            },
            "minProperties": 1,
            "additionalProperties": False,
        }
