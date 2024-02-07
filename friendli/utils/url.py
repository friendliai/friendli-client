# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli CLI URL Utilities."""

from __future__ import annotations

from urllib.parse import urljoin, urlparse

from typing_extensions import deprecated

discuss_url = "https://discuss.friendli.ai/"


def get_host(url: str) -> str:
    """Get a host part of a URL."""
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    netloc = parsed_url.netloc
    return f"{scheme}://{netloc}/"


class URLProvider:
    """Service URL provider."""

    suite_url = ""
    api_url = ""
    training_url = ""
    registry_url = ""
    serving_url = ""
    auth_url = ""
    meter_url = ""
    observatory_url = ""
    web_backend_url = ""

    @classmethod
    def get_suite_uri(cls, path: str) -> str:
        """Get Friendli Suite URI."""
        return urljoin(cls.suite_url, path)

    @classmethod
    def get_auth_uri(cls, path: str) -> str:
        """Get PFA URI."""
        return urljoin(cls.auth_url, path)

    @classmethod
    def get_web_backend_uri(cls, path: str) -> str:
        """Get PF Web Backend API URI."""
        return urljoin(cls.web_backend_url, path)

    @classmethod
    @deprecated("All functionalities to be migrated to get_web_backend_uri")
    def get_training_uri(cls, path: str) -> str:
        """Get PFT URI."""
        return urljoin(cls.training_url, path)

    @classmethod
    def get_api_uri(cls, path: str) -> str:
        """Get PFT URI."""
        return urljoin(cls.api_url, path)

    @classmethod
    def get_serving_uri(cls, path: str) -> str:
        """Get PFS URI."""
        return urljoin(cls.serving_url, path)

    @classmethod
    def get_mr_uri(cls, path: str) -> str:
        """Get PFR URI."""
        return urljoin(cls.registry_url, path)

    @classmethod
    def get_meter_uri(cls, path: str) -> str:
        """Get PFM URI."""
        return urljoin(cls.meter_url, path)

    @classmethod
    def get_observatory_uri(cls, path: str) -> str:
        """Get PFO URI."""
        return urljoin(cls.observatory_url, path)


class ProductionURLProvider(URLProvider):
    """Production service URL provider."""

    suite_url = "https://suite.friendli.ai/"
    registry_url = "https://modelregistry.friendli.ai/"
    serving_url = "https://serving.friendli.ai/"
    auth_url = "https://auth.friendli.ai/"
    meter_url = "https://metering.friendli.ai/"
    observatory_url = "https://observatory.friendli.ai/"
    web_backend_url = "https://suite.friendli.ai/"
    training_url = "https://training.friendli.ai/api/"


class StagingURLProvider(URLProvider):
    """Staging service URL provider."""

    suite_url = "https://suite-staging.friendli.ai/"
    registry_url = "https://pfmodelregistry-staging.friendli.ai/"
    serving_url = "https://pfs-staging.friendli.ai/"
    auth_url = "https://pfauth-staging.friendli.ai/"
    meter_url = "https://pfmeter-staging.friendli.ai/"
    observatory_url = "https://pfo-staging.friendli.ai/"
    web_backend_url = "https://api-staging.friendli.ai/"
    training_url = "https://api-staging.friendli.ai/api/"


class DevURLProvider(URLProvider):
    """Dev service URL provider."""

    suite_url = "https://suite-dev.friendli.ai/"
    registry_url = "https://pfmodelregistry-dev.friendli.ai/"
    serving_url = "https://pfs-dev.friendli.ai/"
    auth_url = "https://pfauth-dev.friendli.ai/"
    meter_url = "https://pfmeter-dev.friendli.ai/"
    observatory_url = "https://pfo-dev.friendli.ai/"
    web_backend_url = "https://api-dev.friendli.ai/"
    training_url = "https://api-dev.friendli.ai/api/"
