# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Tokenization API."""

# pylint: disable=no-name-in-module

from __future__ import annotations

from typing import List

import requests
from pydantic import ValidationError
from requests import HTTPError

from periflow.errors import APIError, SessionClosedError
from periflow.schema.api.v1.codegen.tokenize_pb2 import V1TokenizeRequest
from periflow.schema.api.v1.tokenize import V1TokenizeResponse
from periflow.sdk.api.base import BaseAPI
from periflow.utils.request import DEFAULT_REQ_TIMEOUT


class Tokenization(BaseAPI):
    """PeriFlow tokenization API."""

    @property
    def _api_path(self) -> str:
        return "v1/tokenize"

    def create(self, prompt: str) -> List[int]:
        """Create a tokenization result.

        Args:
            prompt (str): Input prompt in text.

        Raises:
            APIError: Raised when the input has invalid format.

        Returns:
            List[int]: A list of token IDs.

        """
        request_pb = V1TokenizeRequest(prompt=prompt)
        request_data = request_pb.SerializeToString()
        try:
            response = requests.post(
                url=self._endpoint,
                data=request_data,
                headers=self._get_headers(),
                timeout=DEFAULT_REQ_TIMEOUT,
            )
        except HTTPError as exc:
            raise APIError(str(exc)) from exc

        resp_data = response.json()
        try:
            tokenize_res = V1TokenizeResponse.model_validate(resp_data)
        except ValidationError as exc:
            raise APIError(resp_data) from exc

        return tokenize_res.tokens

    async def acreate(self, prompt: str) -> List[int]:
        """Create a tokenization result asynchronously.

        :::info
        You must open API session with `api_session()` before `acreate()`.
        :::

        Args:
            prompt (str): Input prompt in text.

        Raises:
            APIError: Raised when the input has invalid format.

        Returns:
            List[int]: A list of token IDs.

        """
        if self._session is None:
            raise SessionClosedError("Create a session with 'api_session' first.")

        request_pb = V1TokenizeRequest(prompt=prompt)
        request_data = request_pb.SerializeToString()
        response = await self._session.post(url=self._endpoint, data=request_data)

        if 400 <= response.status < 500:
            raise APIError(
                f"{response.status} Client Error: {response.reason} for url: {self._endpoint}"
            )
        if 500 <= response.status < 600:
            raise APIError(
                f"{response.status} Server Error: {response.reason} for url: {self._endpoint}"
            )

        resp_data = await response.json()
        try:
            tokenize_res = V1TokenizeResponse.model_validate(resp_data)
        except ValidationError as exc:
            raise APIError(resp_data) from exc

        return tokenize_res.tokens
