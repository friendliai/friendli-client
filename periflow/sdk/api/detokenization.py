# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Detokenization API."""

# pylint: disable=no-name-in-module

from __future__ import annotations

from typing import List

import requests
from pydantic import ValidationError
from requests import HTTPError

from periflow.errors import APIError, SessionClosedError
from periflow.schema.api.v1.codegen.detokenize_pb2 import V1DetokenizeRequest
from periflow.schema.api.v1.detokenize import V1DetokenizeResponse
from periflow.sdk.api.base import BaseAPI
from periflow.utils.request import DEFAULT_REQ_TIMEOUT


class Detokenization(BaseAPI):
    """PeriFlow detokenization API."""

    @property
    def _api_path(self) -> str:
        return "v1/detokenize"

    def create(self, tokens: List[int]) -> str:
        """Create a detokenization result.

        Args:
            tokens (List[int]): A list of input token IDs.

        Raises:
            APIError: Raised when the input has invalid format.

        Returns:
            str: A detokenized text string.

        """
        request_pb = V1DetokenizeRequest()
        request_pb.tokens.MergeFrom(tokens)  # type: ignore
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
            detokenize_res = V1DetokenizeResponse.model_validate(resp_data)
        except ValidationError as exc:
            raise APIError(resp_data) from exc

        return detokenize_res.text

    async def acreate(self, tokens: List[int]) -> str:
        """Create a detokenization result asynchronously.

        :::info
        You must open API session with `api_session()` before `acreate()`.
        :::

        Args:
            tokens (List[int]): A list of input token IDs.

        Raises:
            APIError: Raised when the input has invalid format.

        Returns:
            str: A detokenized text string.

        """
        if self._session is None:
            raise SessionClosedError("Create a session with 'api_session' first.")

        request_pb = V1DetokenizeRequest()
        request_pb.tokens.MergeFrom(tokens)  # type: ignore
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
            detokenize_res = V1DetokenizeResponse.model_validate(resp_data)
        except ValidationError as exc:
            raise APIError(resp_data) from exc

        return detokenize_res.text
