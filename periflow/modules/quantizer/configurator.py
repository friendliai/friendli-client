# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Quantizer Configurator."""

from __future__ import annotations

import json
from typing import Any, Dict, cast

from periflow.configurator.base import IO, Configurator
from periflow.utils.format import secho_error_and_exit


class QuantConfigurator(Configurator):
    """Configurator for quantization config."""

    @property
    def validation_schema(self) -> Dict[str, Any]:
        """JSON schema to validate quantization configuration contents."""
        return {
            "type": "object",
            "properties": {
                "quant_mode": {"type": "string"},
                "quant_args": {"type": "object"},
            },
            "required": ["quant_mode", "quant_args"],
        }

    @classmethod
    def from_file(cls, f: IO) -> Configurator:
        """Build a DRC configurator from a DRC file."""
        try:
            config: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            secho_error_and_exit(f"Error occurred while parsing config file: {e!r}")

        return cls(config)  # type: ignore


class SmoothQuantConfigurator(QuantConfigurator):
    """Configurator for Smoothquant config."""

    @property
    def validation_schema(self) -> Dict[str, Any]:
        """JSON schema to validate SmoothQuant configuration contents."""
        quant_validation_scheme = super().validation_schema
        cast(Dict, quant_validation_scheme["properties"]["quant_args"]).update(
            {
                "properties": {
                    "data_path_or_name": {"type": "string"},
                    "data_format": {"type": "string"},
                    "data_split": {"type": "string"},
                    "num_samples": {"type": "integer"},
                    "max_length": {"type": "integer"},
                    "device": {"type": "string"},
                    "seed": {"type": "integer"},
                    "migration_strength": {"type": "number"},
                },
                "required": [
                    "data_path_or_name",
                    "data_format",
                    "data_split",
                    "num_samples",
                    "max_length",
                    "device",
                    "seed",
                    "migration_strength",
                ],
            }
        )
        return quant_validation_scheme
