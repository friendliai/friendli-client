# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow Completion API (v1)."""

# pylint: disable=line-too-long

from __future__ import annotations

import json
from typing import Literal, Optional, Union, overload

import requests
from pydantic import ValidationError
from requests import HTTPError

from periflow.errors import APIError, InvalidGenerationError, SessionClosedError
from periflow.schema.api.v1.completion import (
    V1Completion,
    V1CompletionLine,
    V1CompletionOptions,
)
from periflow.sdk.api.base import AsyncGenerationStream, GenerationStream, ServingAPI
from periflow.utils.request import DEFAULT_REQ_TIMEOUT


class Completion(
    ServingAPI[
        V1Completion,
        "V1CompletionStream",
        "V1AsyncCompletionStream",
        V1CompletionOptions,
    ]
):
    """PeriFlow Completion API."""

    @property
    def _api_path(self) -> str:
        return "v1/completions"

    @overload
    def create(
        self, options: V1CompletionOptions, *, stream: Literal[True]
    ) -> V1CompletionStream:
        """[skip-doc]."""

    @overload
    def create(
        self, options: V1CompletionOptions, *, stream: Literal[False]
    ) -> V1Completion:
        """[skip-doc]."""

    def create(
        self, options: V1CompletionOptions, *, stream: bool = False
    ) -> Union[V1CompletionStream, V1Completion]:
        """Creates a completion.

        Args:
            options (V1CompletionOptions): Options for the completion.
            stream (bool, optional): Enables streaming mode. Defaults to False.

        Raises:
            APIError: Raised when the HTTP API request to the deployment fails.

        Returns:
            Union[V1Completion, CompletionStream]: If `stream` is `True`, a `CompletionStream` object that iterates the results per token is returned. Otherwise, a `V1CompletionResult` object is returned.

        """
        options.stream = stream

        try:
            response = requests.post(
                url=self._endpoint,
                json=options.model_dump(),
                headers=self._get_headers(),
                stream=stream,
                timeout=DEFAULT_REQ_TIMEOUT,
            )
            response.raise_for_status()
        except HTTPError as exc:
            raise APIError(str(exc)) from exc

        if stream:
            return V1CompletionStream(response=response)
        return V1Completion.model_validate(response.json())

    @overload
    async def acreate(
        self, options: V1CompletionOptions, *, stream: Literal[True]
    ) -> V1AsyncCompletionStream:
        """[skip-doc]."""

    @overload
    async def acreate(
        self, options: V1CompletionOptions, *, stream: Literal[False]
    ) -> V1Completion:
        """[skip-doc]."""

    async def acreate(
        self, options: V1CompletionOptions, *, stream: bool = False
    ) -> Union[V1AsyncCompletionStream, V1Completion]:
        """Creates a completion.

        :::info
        You must open API session with `api_session()` before `acreate()`.
        :::

        Args:
            options (V1CompletionOptions): Options for the completion.
            stream (bool, optional): When set True, enables streaming mode. Defaults to False.

        Raises:
            APIError: Raised when the HTTP API request to the deployment fails.
            SessionClosedError: Raised when the client session is not opened with `api_session()`.

        Returns:
            Union[V1Completion, AsyncCompletionStream]: If `stream` is `True`, a `AsyncCompletionStream` object that iterates the results per token is returned. Otherwise, a `V1CompletionResult` object is returned.

        Examples:
            Basic usage:

            ```python
            from periflow import Completion, V1CompletionOptions

            api = Completion(deployment_id="periflow-deployment-1b9483a0")
            async with api.api_session():
                completion = await api.acreate(
                    options=V1CompletionOptions(
                        prompt="Python is a popular language for",
                        max_tokens=100,
                        top_p=0.8,
                        temperature=0.5,
                        no_repeat_ngram=3,
                    )
                )

            print(completion.choices[0].text)
            ```

            Usage of streaming mode:

            ```python
            from periflow import Completion, V1CompletionOptions

            api = Completion(deployment_id="periflow-deployment-1b9483a0")
            async with api.api_session():
                astream = await api.acreate(
                    options=V1CompletionOptions(
                        prompt="Python is a popular language for",
                        max_tokens=100,
                        top_p=0.8,
                        temperature=0.5,
                        no_repeat_ngram=3,
                    ),
                    stream=True,
                )

                # Iterate over a generation stream.
                async for line in astream:
                    print(line.text)

                # Or you can wait for the generation stream to complete.
                completion = await astream.wait()
            ```

        """
        options.stream = stream

        if self._session is None:
            raise SessionClosedError("Create a session with 'api_session' first.")

        response = await self._session.post(
            url=self._endpoint, json=options.model_dump()
        )

        if 400 <= response.status < 500:
            raise APIError(
                f"{response.status} Client Error: {response.reason} for url: {self._endpoint}"
            )
        if 500 <= response.status < 600:
            raise APIError(
                f"{response.status} Server Error: {response.reason} for url: {self._endpoint}"
            )

        if stream:
            return V1AsyncCompletionStream(response=response)
        return V1Completion.model_validate(await response.json())


class V1CompletionStream(GenerationStream[V1CompletionLine, V1Completion]):
    """Completion stream."""

    def __next__(self) -> V1CompletionLine:  # noqa: D105
        line: bytes = next(self._iter)
        while not line:
            line = next(self._iter)

        parsed = json.loads(line.decode().strip("data: "))
        try:
            return V1CompletionLine.model_validate(parsed)
        except ValidationError as exc:
            try:
                # The last iteration of the stream returns a response with `V1Completion` schema.
                V1Completion.model_validate(parsed)
                raise StopIteration from exc
            except ValidationError:
                raise InvalidGenerationError(
                    f"Generation result has invalid schema: {str(exc)}"
                ) from exc

    def wait(self) -> Optional[V1Completion]:
        """Waits for the generation to complete.

        Raises:
            InvalidGenerationError: Raised when the generation result has invalid format.

        Returns:
            Optional[V1Completion]: The full generation result.

        """
        for line in self._iter:
            if line:
                parsed = json.loads(line.decode().strip("data: "))
                try:
                    # The last iteration of the stream returns a response with `V1Completion` schema.
                    return V1Completion.model_validate(parsed)
                except ValidationError as exc:
                    try:
                        # Skip the line response.
                        V1CompletionLine.model_validate(parsed)
                    except ValidationError:
                        raise InvalidGenerationError(
                            f"Generation result has invalid schema: {str(exc)}"
                        ) from exc
        return None


class V1AsyncCompletionStream(AsyncGenerationStream[V1CompletionLine, V1Completion]):
    """Asynchronous completion stream."""

    async def __anext__(self) -> V1CompletionLine:  # noqa: D105
        line: bytes = await self._iter.__anext__()
        while not line or line == b"\n":
            line = await self._iter.__anext__()

        parsed = json.loads(line.decode().strip("data: "))
        try:
            return V1CompletionLine.model_validate(parsed)
        except ValidationError as exc:
            try:
                # The last iteration of the stream returns a response with `V1Completion` schema.
                V1Completion.model_validate(parsed)
                raise StopAsyncIteration from exc
            except ValidationError:
                raise InvalidGenerationError(
                    f"Generation result has invalid schema: {str(exc)}"
                ) from exc

    async def wait(self) -> Optional[V1Completion]:  # noqa: D105
        """Waits for the generation to complete.

        Raises:
            InvalidGenerationError: Raised when the generation result has invalid format.

        Returns:
            Optional[V1Completion]: The full generation result.

        """
        async for line in self._iter:
            if line and line != b"\n":
                parsed = json.loads(line.decode().strip("data: "))
                try:
                    # The last iteration of the stream returns a response with `V1Completion` schema.
                    return V1Completion.model_validate(parsed)
                except ValidationError as exc:
                    try:
                        # Skip the line response.
                        V1CompletionLine.model_validate(parsed)
                    except ValidationError:
                        raise InvalidGenerationError(
                            f"Generation result has invalid schema: {str(exc)}"
                        ) from exc
        return None
