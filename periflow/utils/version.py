# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Version utils."""


from __future__ import annotations

import importlib
import json
from importlib.metadata import version
from typing import List
from urllib.request import urlopen

from packaging.version import _BaseVersion
from packaging.version import parse as parse_version

PERIFLOW_PACKAGE_NAME = "periflow-client"
PYPI_BASE_URL = "https://pypi.org/pypi"


def is_latest_version(ver: str) -> bool:
    """Check if the installed CLI version is the latest release."""
    version = parse_version(ver)
    latest_version = get_latest_version()

    return version >= latest_version


def get_latest_version() -> _BaseVersion:
    """Get the latest CLI release version."""
    with urlopen(f"{PYPI_BASE_URL}/{PERIFLOW_PACKAGE_NAME}/json") as resp:
        pypi_info = json.loads(resp.read())
    latest_ver_string = pypi_info["info"]["version"]
    return parse_version(latest_ver_string)


def get_installed_version() -> str:
    """Get the currently installed CLI version."""
    return version(PERIFLOW_PACKAGE_NAME)


def check_dependencies_installed(deps: List[str]) -> bool:
    """Check if dependencies are installed."""
    for dep in deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            return False

    return True


def check_extras_installed() -> bool:
    """Check extra package dependencies are installed."""
    return check_dependencies_installed(["torch", "transformers"])
