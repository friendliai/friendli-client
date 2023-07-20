# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CLI URL Utilities."""

from __future__ import annotations

from urllib.parse import urljoin, urlparse

training_url = "https://training.periflow.ai/api/"
training_ws_url = "wss://training-ws.periflow.ai/ws/"
discuss_url = "https://discuss.friendli.ai/"
registry_url = "https://modelregistry.periflow.ai/"
serving_url = "https://serving.periflow.ai/"
auth_url = "https://auth.periflow.ai/"
meter_url = "https://metering.periflow.ai/"
observatory_url = "https://observatory.periflow.ai/"


def get_auth_uri(path: str) -> str:
    """Get PFA URI."""
    return urljoin(auth_url, path)


def get_training_uri(path: str) -> str:
    """Get PFT URI."""
    return urljoin(training_url, path)


def get_training_ws_uri(path: str) -> str:
    """Get PFT websocket URI."""
    return urljoin(training_ws_url, path)


def get_serving_uri(path: str) -> str:
    """Get PFS URI."""
    return urljoin(serving_url, path)


def get_mr_uri(path: str) -> str:
    """Get PFR URI."""
    return urljoin(registry_url, path)


def get_meter_uri(path: str) -> str:
    """Get PFM URI."""
    return urljoin(meter_url, path)


def get_observatory_uri(path: str) -> str:
    """Get PFO URI."""
    return urljoin(observatory_url, path)


def get_baseurl(url: str) -> str:
    """Get a base of a URL."""
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    base = parsed_url.netloc
    return f"{scheme}://{base}/"
