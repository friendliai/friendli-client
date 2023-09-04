# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CLI Validation Utilities."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from importlib.util import find_spec
from typing import Any, Dict, Optional, Type

import typer
from pydantic import ValidationError

from periflow.cloud.storage import cloud_region_map, storage_region_map
from periflow.enums import CloudType, StorageType
from periflow.errors import (
    InvalidAttributesError,
    InvalidConfigError,
    NotSupportedError,
)
from periflow.schema.resource.v1.attributes import V1AttributesValidationModel
from periflow.utils.format import secho_error_and_exit
from periflow.utils.version import (
    PERIFLOW_PACKAGE_NAME,
    get_installed_version,
    get_latest_version,
    is_latest_version,
)


def validate_storage_region(vendor: StorageType, region: str) -> None:
    """Validation the cloud storage availability region."""
    available_regions = storage_region_map[vendor]
    if region not in available_regions:
        raise InvalidConfigError(
            f"'{region}' is not supported region for {vendor}. "
            f"Please choose another one in {available_regions}."
        )


def validate_cloud_region(vendor: CloudType, region: str) -> None:
    """Validate the cloud availability region."""
    available_regions = cloud_region_map[vendor]
    if region not in available_regions:
        secho_error_and_exit(
            f"'{region}' is not a supported region for {vendor}. "
            f"Please choose another one in {available_regions}."
        )


def validate_datetime_format(datetime_str: Optional[str]) -> Optional[str]:
    """Validate the datetime format."""
    if datetime_str is None:
        return None

    try:
        datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:
        raise typer.BadParameter(
            "The datetime format should be {YYYY}-{MM}-{DD}T{HH}:{MM}:{SS}"
        ) from exc
    return datetime_str


def validate_cloud_storage_type(val: StorageType) -> None:
    """Validate the cloud storage type."""
    if val == StorageType.FAI:
        raise NotSupportedError(
            "Checkpoint creation with FAI storage is not supported now."
        )


def validate_package_version() -> None:
    """Validate the installed CLI version."""
    installed_version = get_installed_version()
    if not is_latest_version(installed_version):
        latest_version = get_latest_version()
        secho_error_and_exit(
            f"Package version({installed_version}) is deprecated. "
            f"Please install the latest version({latest_version}) with "
            f"'pip install {PERIFLOW_PACKAGE_NAME}=={latest_version} -U'."
        )


def validate_checkpoint_attributes(attr: Dict[str, Any]) -> None:
    """Validate checkpoint attributes schema."""
    try:
        V1AttributesValidationModel.model_validate({"attr": attr})
    except ValidationError as exc:
        msgs = []
        for error in exc.errors():
            error_type = error["type"]
            if error_type == "union_tag_invalid":
                msgs.append("'model_type' is not one of the expected values.")
            elif error_type == "union_tag_not_found":
                msgs.append("'model_type' filed is missing.")
            else:
                msgs.append(f"{error['msg']}. Correct the field '{error['loc'][-1]}'")

        if len(msgs) == 1:
            msg = msgs[0]
        else:
            msg = "\n>>> " + "\n>>> ".join(msgs)
        raise InvalidAttributesError(msg) from exc


def validate_enums(val: Any, enum_cls: Type[Enum]) -> Any:
    """Validate if the value is the proper enum."""
    try:
        return enum_cls(val)
    except ValueError as exc:
        supported_values = set([e.value for e in enum_cls])
        raise InvalidConfigError(
            f"Invalid value. Please provide one of {supported_values}"
        ) from exc


def validate_convert_imports() -> None:
    """Validate the import modules for checkpoint conversion."""
    if find_spec("torch") is None:
        raise ModuleNotFoundError(
            "To convert the checkpoint, you must install 'torch'."
        )
    if find_spec("transformers") is None or find_spec("accelerate") is None:
        raise ModuleNotFoundError(
            "To convert the checkpoint,"
            " your must install the package with 'pip install periflow-client[mllib]'"
        )
