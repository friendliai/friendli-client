# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Model spec utils"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import yaml
from jinja2.environment import Template as JinjaTemplate
from pydantic import BaseModel


class InvalidSpecFormatError(Exception):
    """Invalid model spec format  that can be handled by users."""


class SpecNodeType(str, Enum):
    """Model spec node type."""

    DATA = "data"
    GROUP = "group"
    REPEAT_GROUP = "repeat_group"


class ParamInfo(BaseModel):
    """Parameter info."""

    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]

    class Config:
        arbitrary_types_allowed = (
            True  # for np.dtype only check `isinstance(dtype, np.dtype)`
        )

    @classmethod
    def load(cls, name: str, data: Dict[str, Any]) -> ParamInfo:
        """Load a param info from data.

        Args:
            name (str): Name of parameter.
            data (dict[str, Any]): A dictionary describing the parameter info.

        Raises:
            InvalidSpecFormatError: Raised if required key does not exist in data.

        Returns:
            ParamInfo: Loaded param info.

        """
        try:
            dtype = np.dtype(data["dtype"])
            return ParamInfo(
                name=name,
                dtype=dtype,
                shape=tuple(map(int, data["shape"])),
            )
        except (KeyError, AttributeError, TypeError) as exc:
            raise InvalidSpecFormatError from exc


class RepeatRange(BaseModel):
    """Repeat group's repeat range."""

    lo: int
    hi: int


class Template:
    """Renderable YAML template."""

    def __init__(self, jinja_template: JinjaTemplate):
        self._jinja2_template = jinja_template

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Template:
        with open(path, "r") as f:
            return cls(jinja_template=JinjaTemplate(f.read()))

    def render(self, **kwargs) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Render a Jinja2-YAML template with filling the variables.

        Returns:
            dict[str, Any] | list[dict[str, Any]]: Rendered template in JSON format.

        """
        return yaml.safe_load(self._jinja2_template.render(**kwargs))


class ModelSpecParser:
    """Model spec parser"""

    def __init__(self, model_spec: Dict[str, Any]) -> None:
        """Intialize model spec parser.

        Args:
            model_spec (dict[str, Any]): A dictionary describing the entire model spec.

        """
        self._model_spec = model_spec

    def get_all_param_info(self) -> Dict[ParamInfo]:
        """Get all parameter info specified in the model spec.

        Returns:
            list[ParamInfo]: A list of param info.

        """
        return self._get_param_info(self._model_spec)

    def _get_param_info(
        self, spec: Dict[str, Any], name_prefix: str = ""
    ) -> Dict[ParamInfo]:
        """Get a dictionary of param info in recursion.

        Args:
            spec (dict[str, Any]): Full or partial model spec.
            name_prefix (str, optional): Parsed name until the current recursion step. Defaults to "".

        Returns:
            Dict[ParamInfo]: A dictionary of param info.

        """
        try:
            node_type = spec["type"]
        except KeyError as exc:
            raise InvalidSpecFormatError from exc

        if node_type == SpecNodeType.DATA:
            return {name_prefix: ParamInfo.load(name=name_prefix, data=spec)}
        if node_type == SpecNodeType.GROUP:
            res = {}
            for child_name, child_spec in spec.items():
                if child_name == "type":
                    continue
                res.update(
                    self._get_param_info(
                        spec=child_spec,
                        name_prefix=f"{name_prefix}/{child_name}"
                        if name_prefix
                        else child_name,
                    )
                )
            return res
        if node_type == SpecNodeType.REPEAT_GROUP:
            try:
                repeat_range = RepeatRange.model_validate(spec["range"])  # type: ignore
            except KeyError as exc:
                raise InvalidSpecFormatError from exc
            res = {}
            for i in range(repeat_range.lo, repeat_range.hi + 1):
                for child_name, child_spec in spec.items():
                    if child_name in ["type", "range"]:
                        continue
                    res.update(
                        self._get_param_info(
                            spec=child_spec,
                            name_prefix=f"{name_prefix.replace('*', str(i))}/{child_name}"
                            if name_prefix
                            else child_name,
                        )
                    )
            return res
        raise InvalidSpecFormatError
