# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Configurator."""

from __future__ import annotations

import json
from typing import Any, Dict

from periflow.configurator.base import IO, Configurator
from periflow.utils.format import secho_error_and_exit


class SmoothQuantConfigurator(Configurator):
    """Configurator for default request config."""

    @property
    def validation_schema(self) -> dict:
        """JSON schema to validate SmoothQuant configuration contents."""
        return {
            "type": "object",
            "properties": {
                "smoothquant_config": {
                    "type": "object",
                    "properties": {
                        "data_path": {"type": "string"},
                        "data_format": {"type": "string"},
                        "data_type": {"type": "string"},
                        "num_samples": {"type": "integer"},
                        "max_length": {"type": "integer"},
                        "device": {"type": "string"},
                        "seed": {"type": "integer"},
                        "migration_strength": {"type": "number"},
                    },
                }
            },
            "required": ["smoothquant_config"],
        }

    @classmethod
    def from_file(cls, f: IO) -> Configurator:
        """Build a DRC configurator from a DRC file."""
        try:
            config: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            secho_error_and_exit(f"Error occurred while parsing config file: {e!r}")

        return cls(config)  # type: ignore
