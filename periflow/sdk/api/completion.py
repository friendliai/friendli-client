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

        The `options` argument gets a `V1CompletionOptions` object, which has the following schema.

        | Argument | Type | Default |
        |----------|------|---------|
        | `stream` | `Optional[bool]` | `None` |
        | `prompt` | `Optional[str]` | `None` |
        | `tokens` | `Optional[List[int]]` | `None` |
        | `timeout_microseconds` | `Optional[int]` | `None` |
        | `max_tokens` | `Optional[int]` | `None` |
        | `max_total_tokens` | `Optional[int]` | `None` |
        | `min_tokens` | `Optional[int]` | `None` |
        | `min_total_tokens` | `Optional[int]` | `None` |
        | `n` | `Optional[int]` | `None` |
        | `num_beams` | `Optional[int]` | `None` |
        | `length_penalty` | `Optional[float]` | `None` |
        | `early_stopping` | `Optional[bool]` | `None` |
        | `no_repeat_ngram` | `Optional[int]` | `None` |
        | `encoder_no_repeat_ngram` | `Optional[int]` | `None` |
        | `repetition_penalty` | `Optional[float]` | `None` |
        | `encoder_repetition_penalty` | `Optional[float]` | `None` |
        | `temperature` | `Optional[float]` | `None` |
        | `top_k` | `Optional[int]` | `None` |
        | `top_p` | `Optional[float]` | `None` |
        | `stop` | `Optional[List[str]]` | `None` |
        | `stop_tokens` | `Optional[List[TokenSequence]]` <br></br> `(TokenSequence: {"tokens": List[int]})` | `None` |
        | `seed` | `Optional[List[int]]` | `None` |
        | `beam_search_type` | `Optional[BeamSearchType]` | `None` |
        | `beam_compat_pre_normalization` | `Optional[bool]` | `None` |
        | `beam_compat_no_post_normalization` | `Optional[bool]` | `None` |
        | `bad_words` | `Optional[List[str]]` | `None` |
        | `bad_word_tokens` | `Optional[List[TokenSequence]]` <br></br> `(TokenSequence: {"tokens": List[int]})` | `None` |
        | `include_output_logits` | `Optional[bool]` | `None` |
        | `eos_token` | `Optional[List[int]]` | `None` |

        Followings are the descriptions for each field.

        - **stream**: Whether to stream generation result. When set true, each token will be sent as server-sent events once generated. Not supported when using beam search.
        - **prompt**: The prompt (i.e., input text) to generate completion for. Either `prompt` or `tokens` field is required.
        - **tokens**: The tokenized prompt (i.e., input tokens). Either `prompt` or `tokens` field is required.
        - **timeout_microseconds**: Request timeout. Gives the HTTP `429 Too Many Requests` response status code. Default behavior is no timeout.
        - **max_tokens**: The maximum number of tokens to generate. For decoder-only models like GPT, the length of your input tokens plus `max_tokens` should not exceed the model's maximum length (e.g., 2048 for OpenAI GPT-3). For encoder-decoder models like T5 or BlenderBot, `max_tokens` should not exceed the model's maximum output length. This is similar to Hugging Face's [`max_new_tokens`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.max_new_tokens) argument.
        - **max_total_tokens**: The maximum number of tokens including both the generated result and the input tokens. Only allowed for decoder-only models. Only one argument between `max_tokens` and `max_total_tokens` is allowed. Default value is the model's maximum length. This is similar to Hugging Face's [`max_length`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.max_length) argument.
        - **min_tokens**: The minimum number of tokens to generate. Default value is 0. This is similar to Hugging Face's [`min_new_tokens`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.min_new_tokens) argument.
        - **min_total_tokens**: The minimum number of tokens including both the generated result and the input tokens. Only allowed for decoder-only models. Only one argument between `min_tokens` and `min_total_tokens` is allowed. This is similar to Hugging Face's [`min_length`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.min_length) argument.
        - **n**: The number of independently generated results for the prompt. Not supported when using beam search. Defaults to 1. This is similar to Hugging Face's [`num_return_sequences`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.num_return_sequences(int, ) argument.
        - **num_beams**: Number of beams for beam search. Numbers between 1 and 31 (both inclusive) are allowed. Default behavior is no beam search. This is similar to Hugging Face's [`num_beams`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.num_beams) argument.
        - **length_penalty**: Coefficient for exponential length penalty that is used with beam search. Only allowed for beam search. Defaults to 1.0. This is similar to Hugging Face's [`length_penalty`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.length_penalty) argument.
        - **early_stopping**: Whether to stop the beam search when at least `num_beams` beams are finished with the EOS token. Only allowed for beam search. Defaults to false. This is similar to Hugging Face's [`early_stopping`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.early_stopping) argument.
        - **no_repeat_ngram**: If this exceeds 1, every ngram of that size can only occur once among the generated result (plus the input tokens for decoder-only models). 1 means that this mechanism is disabled (i.e., you cannot prevent 1-gram from being generated repeatedly). Defaults to 1. This is similar to Hugging Face's [`no_repeat_ngram_size`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.no_repeat_ngram_size) argument.
        - **encoder_no_repeat_ngram**: If this exceeds 1, every ngram of that size occurring in the input token sequence cannot appear in the generated result. 1 means that this mechanism is disabled (i.e., you cannot prevent 1-gram from being generated repeatedly). Only allowed for encoder-decoder models. Defaults to 1. This is similar to Hugging Face's [`encoder_no_repeat_ngram_size`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.encoder_no_repeat_ngram_size) argument.
        - **repetition_penalty**: Penalizes tokens that have already appeared in the generated result (plus the input tokens for decoder-only models). Should be greater than or equal to 1.0. 1.0 means no penalty. See [Keskar et al., 2019](https://arxiv.org/abs/1909.05858) for more details. This is similar to Hugging Face's [`repetition_penalty`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.repetition_penalty) argument.
        - **encoder_repetition_penalty**: Penalizes tokens that have already appeaared in the input tokens. Should be greater than or equal to 1.0. 1.0 means no penalty. Only allowed for encoder-decoder models. See [Keskar et al., 2019](https://arxiv.org/abs/1909.05858) for more details. This is similar to Hugging Face's [`encoder_repetition_penalty`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.encoder_repetition_penalty) argument.
        - **temperature**: Sampling temperature. Non-zero positive numbers are allowed. Smaller temperature makes the generation result closer to greedy, argmax (i.e., `top_k = 1`) sampling. Defaults to 1.0. This is similar to Hugging Face's [`temperature`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.temperature) argument.
        - **top_k**: The number of highest probability tokens to keep for sampling. Numbers between 0 and the vocab size of the model (both inclusive) are allowed. The default value is 0, which means that the API does not apply top-k filtering. This is similar to Hugging Face's [`top_k`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.top_k) argument.
        - **top_p**: Tokens comprising the top `top_p` probability mass are kept for sampling. Numbers between 0.0 (exclusive) and 1.0 (inclusive) are allowed. Defaults to 1.0. This is similar to Hugging Face's [`top_p`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.top_p) argument.
        - **stop**: When one of the stop phrases appears in the generation result, the API will stop generation. The phrase is included in the generated result. If you are using beam search, all of the active beams should contain the stop phrase to terminate generation. Before checking whether a stop phrase is included in the result, the phrase is converted into tokens. We recommend using `stop_tokens` because it is clearer. For example, after tokenization, phrases "clear" and " clear" can result in different token sequences due to the prepended space character. Defaults to empty list.
        - **stop_tokens**: Same as the above `stop` field, but receives token sequences instead of text phrases. A TokenSequence type is a dict with the key 'tokens' and the value type List[int].
        - **seed**: Seed to control random procedure. If nothing is given, the API generate the seed randomly, use it for sampling, and return the seed along with the generated result. When using the `n` argument, you can pass a list of seed values to control all of the independent generations.
        - **beam_search_type**: Which beam search type to use. `DETERMINISTIC` means the standard, deterministic beam search, which is similar to Hugging Face's [`beam_search`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search). Argmuents for controlling random sampling such as `top_k` and `top_p` are not allowed for this option. `STOCHASTIC` means stochastic beam search (more details in [Kool et al. (2019)](https://proceedings.mlr.press/v97/kool19a.html)). `NAIVE_SAMPLING` is similar to Hugging Face's [`beam_sample`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample). Defaults to `DETERMINISTIC`.
        - **bad_words**: Text phrases that should not be generated. For a bad word phrase that contains N tokens, if the first N-1 tokens appears at the last of the generated result, the logit for the last token of the phrase is set to -inf. We recommend using `bad_word_tokens` because it is clearer (more details in the document for `stop` field). Defaults to empty list.
        - **bad_word_tokens**: Same as the above `bad_words` field, but receives token sequences instead of text phrases. A TokenSequence type is a dict with the key 'tokens' and the value type List[int]. This is similar to Hugging Face's <a href="https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.bad_words_ids(List[List[int]]," target="_top">`bad_word_ids`</a> argument.
        - **include_output_logits**: Whether to include the output logits to the generation output.
        - **eos_token**: A list of endpoint sentence tokens.

        :::note
        #### Compatibility with Hugging Face's beam search

        TL;DR: To mimic the default behavior of Hugging Face's beam search, set `beam_compat_pre_normalization` and `beam_compat_no_post_normalization` to true.

        Our inference API provides following options to match the behavior of Hugging Face's beam search (including both `beam_search` and `beam_sample`).
        In Hugging Face's implementation, normalization of logits occurs _before_ perturbing the logits by applying options like [`no_repeat_ngram_size`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.no_repeat_ngram_size); this makes Hugging Face's beam scoring procedure use unnormalized logprobs.
        While the Hugging Face team provides an option that performs normalization again after the perturbation (see [`renormalize_logits`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.renormalize_logits)), they chose not to change their default behavior — you can find related posts [here](https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14) and [there](https://github.com/huggingface/transformers/pull/19143#issuecomment-1287429478).
        By default, our API applies normalization once after the perturbation.
        If you want to stick to Hugging Face's behavior for compatibility reasons, you should control **beam_compat_pre_normalization** and **beam_compat_no_post_normalization** arguments yourself (both default to false).

        To sum up, you have three choices:

        1. Do not control **beam_compat_pre_normalization** and **beam_compat_no_post_normalization** arguments (i.e., leave both of them as false). Our API will apply normalization once after the perturbation, using the normalized logprobs during beam scoring.
        2. Set both **beam_compat_pre_normalization** and **beam_compat_no_post_normalization** arguments to true. Doing so, the API will mimic Hugging Face's default behavior (i.e., `renormalize_logits=False`), performing normalization once before the perturbation.
        3. Set **beam_compat_pre_normalization** to true (and leave **beam_compat_no_post_normalization** as false). Doing so, the API will mimic Hugging Face's behavior _with_ `renormalize_logits=True`, performing normalization both before and after the perturbation. Note that you cannot set **beam_compat_no_post_normalization** as true when **beam_compat_pre_normalization** is false.
        :::

        :::tip
        When `stream` argument of the `create` method is set as `True`, `stream` of
        `V1Completion` is also set as `True`.
        :::

        Args:
            options (V1CompletionOptions): Options for the completion.
            stream (bool, optional): Enables streaming mode. Defaults to False.

        Raises:
            APIError: Raised when the HTTP API request to the deployment fails.

        Returns:
            Union[V1Completion, CompletionStream]: If `stream` is `True`, a `CompletionStream` object that iterates the results per token is returned. Otherwise, a `V1CompletionResult` object is returned.

        Examples:
            Basic usage:

            ```python
            from periflow import Completion, V1CompletionOptions

            api = Completion(deployment_id="periflow-deployment-1b9483a0")
            completion = api.create(
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
            stream = api.create(
                options=V1CompletionOptions(
                    prompt="Python is a popular language for",
                    max_tokens=100,
                    top_p=0.8,
                    temperature=0.5,
                    no_repeat_ngram=3,
                ),
                stream=True,  # Enable stream mode.
            )

            # Iterate over a generation stream.
            for line in stream:
                print(line.text)

            # Or you can wait for the generation stream to complete.
            completion = stream.wait()
            ```

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

        Args:
            options (V1CompletionOptions): Options for the completion.
            stream (bool, optional): When set True, enables streaming mode. Defaults to False.

        Raises:
            APIError: Raised when the HTTP API request to the deployment fails.
            SessionClosedError: Raised when the client session is not opened with `api_session()`.

        Returns:
            Union[V1Completion, AsyncCompletionStream]: If `stream` is `True`, a `AsyncCompletionStream` object that iterates the results per token is returned. Otherwise, a `V1CompletionResult` object is returned.

        Examples:
            :::info
            You must open API session with `api_session()` before `acreate()`.
            :::
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
