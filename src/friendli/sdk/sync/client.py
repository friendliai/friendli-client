# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Suite Sync SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from httpx import URL, Client, HTTPTransport
from typing_extensions import Self

from ...const import BaseUrl
from ...util.httpx.retry_transport import RetryTransportWrapper
from ..auth import BearerAuth
from .base import SyncClientBase
from .resource import (
    AuthResource,
    EndpointResource,
    ModelResource,
    ProjectResource,
    SystemResource,
    TeamResource,
    UserResource,
)

if TYPE_CHECKING:
    from types import TracebackType

    from httpx._types import AuthTypes


class SyncClient(SyncClientBase):
    """Sync client for Friendli Suite API.

    Args:
        auth (str | None): Authentication token for API requests. Optional.
        base_url (str | httpx.URL | None): Base URL of API. Optional.
        http_client (httpx.Client | None): HTTP client for making API requests.
            Optional.

    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, http_client: Client) -> None: ...

    @overload
    def __init__(
        self,
        *,
        auth: str | AuthTypes | None = None,
        base_url: str | URL | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        http_client: Client | None = None,
        auth: str | AuthTypes | None = None,
        base_url: str | URL | None = None,
    ) -> None:
        """Initialize sync client."""
        http_client = self._initialize_http_client(auth, base_url, http_client)
        super().__init__(http_client=http_client)

        # Common
        self.auth = AuthResource(self)
        self.user = UserResource(self)
        self.system = SystemResource(self)
        self.team = TeamResource(self)

        # Dedicated Endpoints
        self.project = ProjectResource(self)
        self.endpoint = EndpointResource(self)
        self.model = ModelResource(self)

    def refresh_http_client(
        self,
        auth: str | AuthTypes | None = None,
        base_url: str | URL | None = None,
        http_client: Client | None = None,
    ) -> None:
        """Re-initialize http client.

        Args:
            auth (str | None): Authentication token for API requests. Optional.
            base_url (str | httpx.URL | None): Base URL of API. Optional.
            http_client (httpx.Client | None): HTTP client for making API requests.
                Optional.

        """
        self._client.close()
        del self._client
        self._client = self._initialize_http_client(auth, base_url, http_client)

    def __enter__(self) -> Self:
        """Context manager for sync client."""
        self._client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self._client.__exit__(exc_type, exc_val, exc_tb)

    def _initialize_http_client(
        self,
        auth: str | AuthTypes | None = None,
        base_url: str | URL | None = None,
        http_client: Client | None = None,
    ) -> Client:
        if http_client is not None:
            return http_client

        if isinstance(auth, str):
            auth = BearerAuth(auth)

        base_url = base_url or BaseUrl

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "rid": "anti-csrf",
            "st-auth-mode": "header",
            "Connection": "close",
        }
        transport = RetryTransportWrapper(
            HTTPTransport(
                http2=False,
                retries=4,
            )
        )
        return Client(
            base_url=base_url,
            auth=auth,
            http2=True,
            timeout=40,
            headers=headers,
            transport=transport,
        )
