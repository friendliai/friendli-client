# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Completion API."""

# pylint: disable=line-too-long, no-name-in-module

from __future__ import annotations

import json
from typing import List, Literal, Optional, Type, Union, overload

from pydantic import ValidationError

from friendli.errors import InvalidGenerationError
from friendli.schema.api.v1.chat.completions import (
    ChatCompletion,
    ChatCompletionLine,
    MessageParam,
)
from friendli.schema.api.v1.codegen.chat_completions_pb2 import V1ChatCompletionsRequest
from friendli.sdk.api.base import (
    AsyncGenerationStream,
    AsyncServingAPI,
    GenerationStream,
    ServingAPI,
)
from friendli.utils.compat import model_parse


class Completions(ServingAPI[Type[V1ChatCompletionsRequest]]):
    """Friendli completions API."""

    @property
    def _api_path(self) -> str:
        return "v1/chat/completions"

    @property
    def _method(self) -> str:
        return "POST"

    @property
    def _content_type(self) -> str:
        return (
            "application/json"
            if self._deployment_id is None
            else "application/protobuf"
        )

    @property
    def _request_pb_cls(self) -> Type[V1ChatCompletionsRequest]:
        return V1ChatCompletionsRequest

    @overload
    def create(
        self,
        *,
        messages: List[MessageParam],
        stream: Literal[True],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout_microseconds: Optional[int] = None,
    ) -> ChatCompletionStream:
        """[skip-doc]."""

    @overload
    def create(
        self,
        *,
        messages: List[MessageParam],
        stream: Literal[False],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout_microseconds: Optional[int] = None,
    ) -> ChatCompletion:
        """[skip-doc]."""

    def create(
        self,
        *,
        messages: List[MessageParam],
        stream: bool,
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout_microseconds: Optional[int] = None,
    ) -> Union[ChatCompletionStream, ChatCompletion]:
        """Creates a chat completion.

        Args:
            messages (List[MessageParam]): A list of messages comprising the conversation so far.
            stream (bool, optional): Whether to stream generation result. When set true, each token will be sent as server-sent events once generated. Not supported when using beam search.
            model (Optional[str]): Code of the model to use. This argument should be set only for serverless endpoints.
            frequency_penalty (Optional[float], optional): Number between -2.0 and 2.0. Positive values penalizes tokens that have been sampled, taking into account their frequency in the preceding text. This penalization diminishes the model's tendency to reproduce identical lines verbatim. Defaults to None.
            presence_penalty (Optional[float], optional): Number between -2.0 and 2.0. Positive values penalizes tokens that have been sampled at least once in the existing text. Defaults to None.
            max_tokens (Optional[int], optional): The maximum number of tokens to generate. The length of your input tokens plus `max_tokens` should not exceed the model's maximum length (e.g., 2048 for OpenAI GPT-3). Defaults to None.
            n (Optional[int], optional): The number of independently generated results for the prompt. Defaults to None.
            stop (Optional[List[str]], optional): When one of the stop phrases appears in the generation result, the API will stop generation. The phrase is included in the generated result. If you are using beam search, all of the active beams should contain the stop phrase to terminate generation. Before checking whether a stop phrase is included in the result, the phrase is converted into tokens. We recommend using `stop_tokens` because it is clearer. For example, after tokenization, phrases "clear" and " clear" can result in different token sequences due to the prepended space character. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Non-zero positive numbers are allowed. Smaller temperature makes the generation result closer to greedy, argmax (i.e., `top_k = 1`) sampling. If it is `None`, then 1.0 is used by default. This is similar to Hugging Face's [`temperature`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.temperature) argument. Defaults to None.
            top_p (Optional[float], optional): Tokens comprising the top `top_p` probability mass are kept for sampling. Numbers between 0.0 (exclusive) and 1.0 (inclusive) are allowed. If it is `None`, then 1.0 is used by default. This is similar to Hugging Face's [`top_p`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.top_p) argument. Defaults to None.
            timeout_microseconds (Optional[int], optional): Request timeout. Gives the HTTP `429 Too Many Requests` response status code. Default behavior is no timeout. Defaults to None.

        Returns:
            Union[ChatCompletionStream, ChatCompletion]: If `stream` is `True`, then `ChatCompletionStream` object that iterates the results per token is returned. Otherwise, a `ChatCompletion` object is returend.

        """
        request_dict = {
            "messages": messages,
            "stream": stream,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "n": n,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "timeout_microseconds": timeout_microseconds,
        }
        response = self._request(data=request_dict, stream=stream, model=model)

        if stream:
            return ChatCompletionStream(response=response)
        return model_parse(ChatCompletion, response.json())


class AsyncCompletions(AsyncServingAPI[Type[V1ChatCompletionsRequest]]):
    """Async completions."""

    @property
    def _api_path(self) -> str:
        return "v1/chat/completions"

    @property
    def _method(self) -> str:
        return "POST"

    @property
    def _content_type(self) -> str:
        return (
            "application/json"
            if self._deployment_id is None
            else "application/protobuf"
        )

    @property
    def _request_pb_cls(self) -> Type[V1ChatCompletionsRequest]:
        return V1ChatCompletionsRequest

    @overload
    async def create(
        self,
        *,
        messages: List[MessageParam],
        stream: Literal[True],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout_microseconds: Optional[int] = None,
    ) -> AsyncChatCompletionStream:
        """[skip-doc]."""

    @overload
    async def create(
        self,
        *,
        messages: List[MessageParam],
        stream: Literal[False],
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout_microseconds: Optional[int] = None,
    ) -> ChatCompletion:
        """[skip-doc]."""

    async def create(
        self,
        *,
        messages: List[MessageParam],
        stream: bool,
        model: Optional[str] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout_microseconds: Optional[int] = None,
    ) -> Union[AsyncChatCompletionStream, ChatCompletion]:
        """Creates a completion asynchronously.

        Args:
            messages (List[MessageParam]): A list of messages comprising the conversation so far.
            stream (bool, optional): Whether to stream generation result. When set true, each token will be sent as server-sent events once generated. Not supported when using beam search.
            model (Optional[str]): Code of the model to use. This argument should be set only for serverless endpoints.
            frequency_penalty (Optional[float], optional): Number between -2.0 and 2.0. Positive values penalizes tokens that have been sampled, taking into account their frequency in the preceding text. This penalization diminishes the model's tendency to reproduce identical lines verbatim. Defaults to None.
            presence_penalty (Optional[float], optional): Number between -2.0 and 2.0. Positive values penalizes tokens that have been sampled at least once in the existing text. Defaults to None.
            max_tokens (Optional[int], optional): The maximum number of tokens to generate. The length of your input tokens plus `max_tokens` should not exceed the model's maximum length (e.g., 2048 for OpenAI GPT-3). Defaults to None.
            n (Optional[int], optional): The number of independently generated results for the prompt. Defaults to None.
            stop (Optional[List[str]], optional): When one of the stop phrases appears in the generation result, the API will stop generation. The phrase is included in the generated result. If you are using beam search, all of the active beams should contain the stop phrase to terminate generation. Before checking whether a stop phrase is included in the result, the phrase is converted into tokens. We recommend using `stop_tokens` because it is clearer. For example, after tokenization, phrases "clear" and " clear" can result in different token sequences due to the prepended space character. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Non-zero positive numbers are allowed. Smaller temperature makes the generation result closer to greedy, argmax (i.e., `top_k = 1`) sampling. If it is `None`, then 1.0 is used by default. This is similar to Hugging Face's [`temperature`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.temperature) argument. Defaults to None.
            top_p (Optional[float], optional): Tokens comprising the top `top_p` probability mass are kept for sampling. Numbers between 0.0 (exclusive) and 1.0 (inclusive) are allowed. If it is `None`, then 1.0 is used by default. This is similar to Hugging Face's [`top_p`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.top_p) argument. Defaults to None.
            timeout_microseconds (Optional[int], optional): Request timeout. Gives the HTTP `429 Too Many Requests` response status code. Default behavior is no timeout. Defaults to None.

        Returns:
            Union[AsyncChatCompletionStream, ChatCompletion]: If `stream` is `True`, then `AsyncChatCompletionStream` object that iterates the results per token is returned. Otherwise, a `ChatCompletion` object is returend.

        """
        request_dict = {
            "messages": messages,
            "stream": stream,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "n": n,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "timeout_microseconds": timeout_microseconds,
        }
        response = await self._request(data=request_dict, stream=stream, model=model)

        if stream:
            return AsyncChatCompletionStream(response=response)
        return model_parse(ChatCompletion, response.json())


class ChatCompletionStream(GenerationStream[ChatCompletionLine]):
    """Completion stream."""

    def __next__(self) -> ChatCompletionLine:  # noqa: D105
        line = next(self._iter)
        while not line:
            line = next(self._iter)

        data = line.strip("data: ")
        if data == "[DONE]":
            raise StopIteration
        parsed = json.loads(data)

        try:
            return model_parse(ChatCompletionLine, parsed)
        except ValidationError as exc:
            raise InvalidGenerationError(
                f"Generation result has invalid schema: {str(exc)}"
            ) from exc


class AsyncChatCompletionStream(AsyncGenerationStream[ChatCompletionLine]):
    """Asynchronous completion stream."""

    async def __anext__(self) -> ChatCompletionLine:  # noqa: D105
        line = await self._iter.__anext__()
        while not line:
            line = await self._iter.__anext__()

        data = line.strip("data: ")
        if data == "[DONE]":
            raise StopAsyncIteration
        parsed = json.loads(data)

        try:
            return model_parse(ChatCompletionLine, parsed)
        except ValidationError as exc:
            raise InvalidGenerationError(
                f"Generation result has invalid schema: {str(exc)}"
            ) from exc
