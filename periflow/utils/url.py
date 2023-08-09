# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CLI URL Utilities."""

from __future__ import annotations

from urllib.parse import urljoin, urlparse

discuss_url = "https://discuss.friendli.ai/"


def get_baseurl(url: str) -> str:
    """Get a base of a URL."""
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    base = parsed_url.netloc
    return f"{scheme}://{base}/"


class URLProvider:
    """Service URL provider."""

    @classmethod
    def get_auth_uri(cls, path: str) -> str:
        """Get PFA URI."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def get_training_uri(cls, path: str) -> str:
        """Get PFT URI."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def get_training_ws_uri(cls, path: str) -> str:
        """Get PFT websocket URI."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def get_serving_uri(cls, path: str) -> str:
        """Get PFS URI."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def get_mr_uri(cls, path: str) -> str:
        """Get PFR URI."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def get_meter_uri(cls, path: str) -> str:
        """Get PFM URI."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def get_observatory_uri(cls, path: str) -> str:
        """Get PFO URI."""
        raise NotImplementedError  # pragma: no cover


class ProductionURLProvider(URLProvider):
    """Production service URL provider."""

    training_url = "https://training.periflow.ai/api/"
    training_ws_url = "wss://training-ws.periflow.ai/ws/"
    registry_url = "https://modelregistry.periflow.ai/"
    serving_url = "https://serving.periflow.ai/"
    auth_url = "https://auth.periflow.ai/"
    meter_url = "https://metering.periflow.ai/"
    observatory_url = "https://observatory.periflow.ai/"

    @classmethod
    def get_auth_uri(cls, path: str) -> str:
        """Get PFA URI."""
        return urljoin(cls.auth_url, path)

    @classmethod
    def get_training_uri(cls, path: str) -> str:
        """Get PFT URI."""
        return urljoin(cls.training_url, path)

    @classmethod
    def get_training_ws_uri(cls, path: str) -> str:
        """Get PFT websocket URI."""
        return urljoin(cls.training_ws_url, path)

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


class StagingURLProvider(URLProvider):
    """Staging service URL provider."""

    training_url = "https://api-staging.periflow.ai/api/"
    training_ws_url = "wss://api-ws-staging.periflow.ai/ws/"
    registry_url = "https://pfmodelregistry-staging.periflow.ai/"
    serving_url = "https://pfs-staging.periflow.ai/"
    auth_url = "https://pfauth-staging.periflow.ai/"
    meter_url = "https://pfmeter-staging.periflow.ai/"
    observatory_url = "https://pfo-staging.periflow.ai/"

    @classmethod
    def get_auth_uri(cls, path: str) -> str:
        """Get PFA URI."""
        return urljoin(cls.auth_url, path)

    @classmethod
    def get_training_uri(cls, path: str) -> str:
        """Get PFT URI."""
        return urljoin(cls.training_url, path)

    @classmethod
    def get_training_ws_uri(cls, path: str) -> str:
        """Get PFT websocket URI."""
        return urljoin(cls.training_ws_url, path)

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


class DevURLProvider(URLProvider):
    """Dev service URL provider."""

    training_url = "https://api-dev.periflow.ai/api/"
    training_ws_url = "wss://api-ws-dev.periflow.ai/ws/"
    registry_url = "https://pfmodelregistry-dev.periflow.ai/"
    serving_url = "https://pfs-dev.periflow.ai/"
    auth_url = "https://pfauth-dev.periflow.ai/"
    meter_url = "https://pfmeter-dev.periflow.ai/"
    observatory_url = "https://pfo-dev.periflow.ai/"

    @classmethod
    def get_auth_uri(cls, path: str) -> str:
        """Get PFA URI."""
        return urljoin(cls.auth_url, path)

    @classmethod
    def get_training_uri(cls, path: str) -> str:
        """Get PFT URI."""
        return urljoin(cls.training_url, path)

    @classmethod
    def get_training_ws_uri(cls, path: str) -> str:
        """Get PFT websocket URI."""
        return urljoin(cls.training_ws_url, path)

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
