# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Completion API."""

# pylint: disable=line-too-long, no-name-in-module, too-many-locals

from __future__ import annotations

import json
from typing import List, Literal, Optional, Type, Union, overload

from pydantic import ValidationError

from friendli.errors import InvalidGenerationError
from friendli.schema.api.v1.codegen.completions_pb2 import V1CompletionsRequest
from friendli.schema.api.v1.completions import (
    BeamSearchType,
    Completion,
    CompletionLine,
    TokenSequenceParam,
)
from friendli.sdk.api.base import (
    AsyncGenerationStream,
    AsyncServingAPI,
    GenerationStream,
    ServingAPI,
)
from friendli.utils.compat import model_parse


class Completions(ServingAPI[Type[V1CompletionsRequest]]):
    """Friendli completions API."""

    @property
    def _api_path(self) -> str:
        return "v1/completions"

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
    def _request_pb_cls(self) -> Type[V1CompletionsRequest]:
        return V1CompletionsRequest

    @overload
    def create(
        self,
        *,
        stream: Literal[True],
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        tokens: Optional[List[int]] = None,
        timeout_microseconds: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        min_total_tokens: Optional[int] = None,
        n: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram: Optional[int] = None,
        encoder_no_repeat_ngram: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        encoder_repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stop_tokens: Optional[List[TokenSequenceParam]] = None,
        seed: Optional[List[int]] = None,
        token_index_to_replace: Optional[List[int]] = None,
        embedding_to_replace: Optional[List[float]] = None,
        beam_search_type: Optional[BeamSearchType] = None,
        beam_compat_pre_normalization: Optional[bool] = None,
        beam_compat_no_post_normalization: Optional[bool] = None,
        bad_words: Optional[List[str]] = None,
        bad_word_tokens: Optional[List[TokenSequenceParam]] = None,
        include_output_logits: Optional[bool] = None,
        include_output_logprobs: Optional[bool] = None,
        forced_output_tokens: Optional[List[int]] = None,
        eos_token: Optional[List[int]] = None,
    ) -> CompletionStream:
        """[skip-doc]."""

    @overload
    def create(
        self,
        *,
        stream: Literal[False],
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        tokens: Optional[List[int]] = None,
        timeout_microseconds: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        min_total_tokens: Optional[int] = None,
        n: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram: Optional[int] = None,
        encoder_no_repeat_ngram: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        encoder_repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stop_tokens: Optional[List[TokenSequenceParam]] = None,
        seed: Optional[List[int]] = None,
        token_index_to_replace: Optional[List[int]] = None,
        embedding_to_replace: Optional[List[float]] = None,
        beam_search_type: Optional[BeamSearchType] = None,
        beam_compat_pre_normalization: Optional[bool] = None,
        beam_compat_no_post_normalization: Optional[bool] = None,
        bad_words: Optional[List[str]] = None,
        bad_word_tokens: Optional[List[TokenSequenceParam]] = None,
        include_output_logits: Optional[bool] = None,
        include_output_logprobs: Optional[bool] = None,
        forced_output_tokens: Optional[List[int]] = None,
        eos_token: Optional[List[int]] = None,
    ) -> Completion:
        """[skip-doc]."""

    def create(
        self,
        *,
        stream: bool,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        tokens: Optional[List[int]] = None,
        timeout_microseconds: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        min_total_tokens: Optional[int] = None,
        n: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram: Optional[int] = None,
        encoder_no_repeat_ngram: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        encoder_repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stop_tokens: Optional[List[TokenSequenceParam]] = None,
        seed: Optional[List[int]] = None,
        token_index_to_replace: Optional[List[int]] = None,
        embedding_to_replace: Optional[List[float]] = None,
        beam_search_type: Optional[BeamSearchType] = None,
        beam_compat_pre_normalization: Optional[bool] = None,
        beam_compat_no_post_normalization: Optional[bool] = None,
        bad_words: Optional[List[str]] = None,
        bad_word_tokens: Optional[List[TokenSequenceParam]] = None,
        include_output_logits: Optional[bool] = None,
        include_output_logprobs: Optional[bool] = None,
        forced_output_tokens: Optional[List[int]] = None,
        eos_token: Optional[List[int]] = None,
    ) -> Union[CompletionStream, Completion]:
        """Creates a completion.

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

        Args:
            stream (bool, optional): Whether to stream generation result. When set true, each token will be sent as server-sent events once generated. Not supported when using beam search.
            model (Optional[str]): ID of the model to use. This argument should be set only for serverless endpoints.
            prompt (Optional[str], optional): The prompt (i.e., input text) to generate completion for. Either `prompt` or `tokens` field is required. Defaults to None.
            tokens (Optional[List[int]], optional): The tokenized prompt (i.e., input tokens). Either `prompt` or `tokens` field is required. Defaults to None.
            timeout_microseconds (Optional[int], optional): Request timeout. Gives the HTTP `429 Too Many Requests` response status code. Default behavior is no timeout. Defaults to None.
            max_tokens (Optional[int], optional): The maximum number of tokens to generate. For decoder-only models like GPT, the length of your input tokens plus `max_tokens` should not exceed the model's maximum length (e.g., 2048 for OpenAI GPT-3). For encoder-decoder models like T5 or BlenderBot, `max_tokens` should not exceed the model's maximum output length. This is similar to Hugging Face's [`max_new_tokens`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.max_new_tokens) argument. Defaults to None.
            max_total_tokens (Optional[int], optional): The maximum number of tokens including both the generated result and the input tokens. Only allowed for decoder-only models. Only one argument between `max_tokens` and `max_total_tokens` is allowed. Default value is the model's maximum length. This is similar to Hugging Face's [`max_length`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.max_length) argument. Defaults to None.
            min_tokens (Optional[int], optional): The minimum number of tokens to generate. Default value is 0. This is similar to Hugging Face's [`min_new_tokens`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.min_new_tokens) argument. Defaults to None.
            min_total_tokens (Optional[int], optional): The minimum number of tokens including both the generated result and the input tokens. Only allowed for decoder-only models. Only one argument between `min_tokens` and `min_total_tokens` is allowed. This is similar to Hugging Face's [`min_length`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.min_length) argument. Defaults to None.
            n (Optional[int], optional): The number of independently generated results for the prompt. Not supported when using beam search. This is similar to Hugging Face's [`num_return_sequences`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.num_return_sequences(int, ) argument. Defaults to None.
            num_beams (Optional[int], optional): Number of beams for beam search. Numbers between 1 and 31 (both inclusive) are allowed. Default behavior is no beam search. This is similar to Hugging Face's [`num_beams`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.num_beams) argument. Defaults to None.
            length_penalty (Optional[float], optional): Coefficient for exponential length penalty that is used with beam search. Only allowed for beam search. Defaults to 1.0. This is similar to Hugging Face's [`length_penalty`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.length_penalty) argument. Defaults to None.
            early_stopping (Optional[bool], optional): Whether to stop the beam search when at least `num_beams` beams are finished with the EOS token. Only allowed for beam search. Defaults to false. This is similar to Hugging Face's [`early_stopping`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.early_stopping) argument. Defaults to None.
            no_repeat_ngram (Optional[int], optional): If this exceeds 1, every ngram of that size can only occur once among the generated result (plus the input tokens for decoder-only models). 1 means that this mechanism is disabled (i.e., you cannot prevent 1-gram from being generated repeatedly). Defaults to 1. This is similar to Hugging Face's [`no_repeat_ngram_size`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.no_repeat_ngram_size) argument. Defaults to None.
            encoder_no_repeat_ngram (Optional[int], optional): If this exceeds 1, every ngram of that size occurring in the input token sequence cannot appear in the generated result. 1 means that this mechanism is disabled (i.e., you cannot prevent 1-gram from being generated repeatedly). Only allowed for encoder-decoder models. Defaults to 1. This is similar to Hugging Face's [`encoder_no_repeat_ngram_size`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.encoder_no_repeat_ngram_size) argument. Defaults to None.
            repetition_penalty (Optional[float], optional): Penalizes tokens that have already appeared in the generated result (plus the input tokens for decoder-only models). Should be greater than or equal to 1.0. 1.0 means no penalty. See [Keskar et al., 2019](https://arxiv.org/abs/1909.05858) for more details. This is similar to Hugging Face's [`repetition_penalty`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.repetition_penalty) argument. Defaults to None.
            encoder_repetition_penalty (Optional[float], optional): Penalizes tokens that have already appeaared in the input tokens. Should be greater than or equal to 1.0. 1.0 means no penalty. Only allowed for encoder-decoder models. See [Keskar et al., 2019](https://arxiv.org/abs/1909.05858) for more details. This is similar to Hugging Face's [`encoder_repetition_penalty`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.encoder_repetition_penalty) argument. Defaults to None.
            frequency_penalty (Optional[float], optional): Number between -2.0 and 2.0. Positive values penalizes tokens that have been sampled, taking into account their frequency in the preceding text. This penalization diminishes the model's tendency to reproduce identical lines verbatim. Defaults to None.
            presence_penalty (Optional[float], optional): Number between -2.0 and 2.0. Positive values penalizes tokens that have been sampled at least once in the existing text. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Non-zero positive numbers are allowed. Smaller temperature makes the generation result closer to greedy, argmax (i.e., `top_k = 1`) sampling. Defaults to 1.0. This is similar to Hugging Face's [`temperature`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.temperature) argument. Defaults to None.
            top_k (Optional[int], optional): The number of highest probability tokens to keep for sampling. Numbers between 0 and the vocab size of the model (both inclusive) are allowed. The default value is 0, which means that the API does not apply top-k filtering. This is similar to Hugging Face's [`top_k`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.top_k) argument. Defaults to None.
            top_p (Optional[float], optional): Tokens comprising the top `top_p` probability mass are kept for sampling. Numbers between 0.0 (exclusive) and 1.0 (inclusive) are allowed. Defaults to 1.0. This is similar to Hugging Face's [`top_p`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.top_p) argument. Defaults to None.
            stop (Optional[List[str]], optional): When one of the stop phrases appears in the generation result, the API will stop generation. The phrase is included in the generated result. If you are using beam search, all of the active beams should contain the stop phrase to terminate generation. Before checking whether a stop phrase is included in the result, the phrase is converted into tokens. We recommend using `stop_tokens` because it is clearer. For example, after tokenization, phrases "clear" and " clear" can result in different token sequences due to the prepended space character. Defaults to None.
            stop_tokens (Optional[List[TokenSequenceParam]], optional): Same as the above `stop` field, but receives token sequences instead of text phrases. A TokenSequence type is a dict with the key 'tokens' and the value type List[int]. Defaults to None.
            seed (Optional[List[int]], optional): Seed to control random procedure. If nothing is given, the API generate the seed randomly, use it for sampling, and return the seed along with the generated result. When using the `n` argument, you can pass a list of seed values to control all of the independent generations. Defaults to None.
            token_index_to_replace (Optional[List[int]], optional): A list of token indices where to replace the embeddings of input tokens provided via either `tokens` or `prompt`. Defaults to None.
            embedding_to_replace (Optional[List[float]], optional): A list of flattened embedding vectors used for replacing the tokens at the specified indices provided via `token_index_to_replace`. Defaults to None.
            beam_search_type (Optional[BeamSearchType], optional): Which beam search type to use. `DETERMINISTIC` means the standard, deterministic beam search, which is similar to Hugging Face's [`beam_search`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search). Argmuents for controlling random sampling such as `top_k` and `top_p` are not allowed for this option. `STOCHASTIC` means stochastic beam search (more details in [Kool et al. (2019)](https://proceedings.mlr.press/v97/kool19a.html)). `NAIVE_SAMPLING` is similar to Hugging Face's [`beam_sample`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample). Defaults to `DETERMINISTIC`. Defaults to None.
            beam_compat_pre_normalization (Optional[bool], optional): Whether to perform normalization both before and after the perturbation. Defaults to None.
            beam_compat_no_post_normalization (Optional[bool], optional): When sets to `True`, do not perform normalization after the perturbation. Defaults to None.
            bad_words (Optional[List[str]], optional): Text phrases that should not be generated. For a bad word phrase that contains N tokens, if the first N-1 tokens appears at the last of the generated result, the logit for the last token of the phrase is set to -inf. We recommend using `bad_word_tokens` because it is clearer (more details in the document for `stop` field). Defaults to empty list. Defaults to None.
            bad_word_tokens (Optional[List[TokenSequenceParam]], optional): Same as the above `bad_words` field, but receives token sequences instead of text phrases. A TokenSequence type is a dict with the key 'tokens' and the value type List[int]. This is similar to Hugging Face's <a href="https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.bad_words_ids(List[List[int]]," target="_top">`bad_word_ids`</a> argument. Defaults to None.
            include_output_logits (Optional[bool], optional): Whether to include the output logits to the generation output. Defaults to None.
            include_output_logprobs (Optional[bool], optional): Whether to include the output logprobs to the generation output. Defaults to None.
            forced_output_tokens (Optional[List[int]], optional): A token sequence that is enforced as a generation output. This option can be used when evaluating the model for the datasets with multi-choice problems (e.g., [HellaSwag](https://huggingface.co/datasets/hellaswag), [MMLU](https://huggingface.co/datasets/cais/mmlu)). Use this option with `include_output_logprobs` to get logprobs for the evaluation. Defaults to None.
            eos_token (Optional[List[int]], optional): A list of endpoint sentence tokens. Defaults to None.

        Raises:
            APIError: Raised when the HTTP API request to the deployment fails.

        Returns:
            Union[Completion, CompletionStream]: If `stream` is `True`, a `CompletionStream` object that iterates the results per token is returned. Otherwise, a `Completion` object is returned.

        Examples:
            Basic usage:

            ```python
            from friendli import Friendli

            client = Friendli(deployment_id="friendli-deployment-1b9483a0")
            completion = client.completions.create(
                prompt="Python is a popular language for",
                stream=False,
                max_tokens=100,
                top_p=0.8,
                temperature=0.5,
                no_repeat_ngram=3,
            )
            print(completion.choices[0].text)
            ```

            Usage of streaming mode:

            ```python
            from friendli import Friendli

            client = Friendli(deployment_id="friendli-deployment-1b9483a0")
            stream = client.completions.create(
                prompt="Python is a popular language for",
                stream=True,  # Enable stream mode.
                max_tokens=100,
                top_p=0.8,
                temperature=0.5,
                no_repeat_ngram=3,
            )

            # Iterate over a generation stream.
            for line in stream:
                print(line.text)

            # Or you can wait for the generation stream to complete.
            completion = stream.wait()
            ```

        """
        request_dict = {
            "stream": stream,
            "prompt": prompt,
            "tokens": tokens,
            "timeout_microseconds": timeout_microseconds,
            "max_tokens": max_tokens,
            "max_total_tokens": max_total_tokens,
            "min_tokens": min_tokens,
            "min_total_tokens": min_total_tokens,
            "n": n,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
            "no_repeat_ngram": no_repeat_ngram,
            "encoder_no_repeat_ngram": encoder_no_repeat_ngram,
            "repetition_penalty": repetition_penalty,
            "encoder_repetition_penalty": encoder_repetition_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop": stop,
            "stop_tokens": stop_tokens,
            "seed": seed,
            "token_index_to_replace": token_index_to_replace,
            "embedding_to_replace": embedding_to_replace,
            "beam_search_type": beam_search_type,
            "beam_compat_pre_normalization": beam_compat_pre_normalization,
            "beam_compat_no_post_normalization": beam_compat_no_post_normalization,
            "bad_words": bad_words,
            "bad_word_tokens": bad_word_tokens,
            "include_output_logits": include_output_logits,
            "include_output_logprobs": include_output_logprobs,
            "forced_output_tokens": forced_output_tokens,
            "eos_token": eos_token,
        }
        response = self._request(data=request_dict, stream=stream, model=model)

        if stream:
            return CompletionStream(response=response)
        return model_parse(Completion, response.json())


class AsyncCompletions(AsyncServingAPI[Type[V1CompletionsRequest]]):
    """Async completions."""

    @property
    def _api_path(self) -> str:
        return "v1/completions"

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
    def _request_pb_cls(self) -> Type[V1CompletionsRequest]:
        return V1CompletionsRequest

    @overload
    async def create(
        self,
        *,
        stream: Literal[True],
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        tokens: Optional[List[int]] = None,
        timeout_microseconds: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        min_total_tokens: Optional[int] = None,
        n: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram: Optional[int] = None,
        encoder_no_repeat_ngram: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        encoder_repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stop_tokens: Optional[List[TokenSequenceParam]] = None,
        seed: Optional[List[int]] = None,
        token_index_to_replace: Optional[List[int]] = None,
        embedding_to_replace: Optional[List[float]] = None,
        beam_search_type: Optional[BeamSearchType] = None,
        beam_compat_pre_normalization: Optional[bool] = None,
        beam_compat_no_post_normalization: Optional[bool] = None,
        bad_words: Optional[List[str]] = None,
        bad_word_tokens: Optional[List[TokenSequenceParam]] = None,
        include_output_logits: Optional[bool] = None,
        include_output_logprobs: Optional[bool] = None,
        forced_output_tokens: Optional[List[int]] = None,
        eos_token: Optional[List[int]] = None,
    ) -> AsyncCompletionStream:
        """[skip-doc]."""

    @overload
    async def create(
        self,
        *,
        stream: Literal[False],
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        tokens: Optional[List[int]] = None,
        timeout_microseconds: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        min_total_tokens: Optional[int] = None,
        n: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram: Optional[int] = None,
        encoder_no_repeat_ngram: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        encoder_repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stop_tokens: Optional[List[TokenSequenceParam]] = None,
        seed: Optional[List[int]] = None,
        token_index_to_replace: Optional[List[int]] = None,
        embedding_to_replace: Optional[List[float]] = None,
        beam_search_type: Optional[BeamSearchType] = None,
        beam_compat_pre_normalization: Optional[bool] = None,
        beam_compat_no_post_normalization: Optional[bool] = None,
        bad_words: Optional[List[str]] = None,
        bad_word_tokens: Optional[List[TokenSequenceParam]] = None,
        include_output_logits: Optional[bool] = None,
        include_output_logprobs: Optional[bool] = None,
        forced_output_tokens: Optional[List[int]] = None,
        eos_token: Optional[List[int]] = None,
    ) -> Completion:
        """[skip-doc]."""

    async def create(
        self,
        *,
        stream: bool = False,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        tokens: Optional[List[int]] = None,
        timeout_microseconds: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        min_total_tokens: Optional[int] = None,
        n: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram: Optional[int] = None,
        encoder_no_repeat_ngram: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        encoder_repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stop_tokens: Optional[List[TokenSequenceParam]] = None,
        seed: Optional[List[int]] = None,
        token_index_to_replace: Optional[List[int]] = None,
        embedding_to_replace: Optional[List[float]] = None,
        beam_search_type: Optional[BeamSearchType] = None,
        beam_compat_pre_normalization: Optional[bool] = None,
        beam_compat_no_post_normalization: Optional[bool] = None,
        bad_words: Optional[List[str]] = None,
        bad_word_tokens: Optional[List[TokenSequenceParam]] = None,
        include_output_logits: Optional[bool] = None,
        include_output_logprobs: Optional[bool] = None,
        forced_output_tokens: Optional[List[int]] = None,
        eos_token: Optional[List[int]] = None,
    ) -> Union[AsyncCompletionStream, Completion]:
        """Creates a completion asynchronously.

        Args:
            stream (bool, optional): Whether to stream generation result. When set true, each token will be sent as server-sent events once generated. Not supported when using beam search. Defaults to False.
            model (Optional[str]): ID of the model to use. This argument should be set only for serverless endpoints.
            prompt (Optional[str], optional): The prompt (i.e., input text) to generate completion for. Either `prompt` or `tokens` field is required. Defaults to None.
            tokens (Optional[List[int]], optional): The tokenized prompt (i.e., input tokens). Either `prompt` or `tokens` field is required. Defaults to None.
            timeout_microseconds (Optional[int], optional): Request timeout. Gives the HTTP `429 Too Many Requests` response status code. Default behavior is no timeout. Defaults to None.
            max_tokens (Optional[int], optional): The maximum number of tokens to generate. For decoder-only models like GPT, the length of your input tokens plus `max_tokens` should not exceed the model's maximum length (e.g., 2048 for OpenAI GPT-3). For encoder-decoder models like T5 or BlenderBot, `max_tokens` should not exceed the model's maximum output length. This is similar to Hugging Face's [`max_new_tokens`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.max_new_tokens) argument. Defaults to None.
            max_total_tokens (Optional[int], optional): The maximum number of tokens including both the generated result and the input tokens. Only allowed for decoder-only models. Only one argument between `max_tokens` and `max_total_tokens` is allowed. Default value is the model's maximum length. This is similar to Hugging Face's [`max_length`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.max_length) argument. Defaults to None.
            min_tokens (Optional[int], optional): The minimum number of tokens to generate. Default value is 0. This is similar to Hugging Face's [`min_new_tokens`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.min_new_tokens) argument. Defaults to None.
            min_total_tokens (Optional[int], optional): The minimum number of tokens including both the generated result and the input tokens. Only allowed for decoder-only models. Only one argument between `min_tokens` and `min_total_tokens` is allowed. This is similar to Hugging Face's [`min_length`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.min_length) argument. Defaults to None.
            n (Optional[int], optional): The number of independently generated results for the prompt. Not supported when using beam search. Defaults to 1. This is similar to Hugging Face's [`num_return_sequences`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.num_return_sequences(int, ) argument. Defaults to None.
            num_beams (Optional[int], optional): Number of beams for beam search. Numbers between 1 and 31 (both inclusive) are allowed. Default behavior is no beam search. This is similar to Hugging Face's [`num_beams`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.num_beams) argument. Defaults to None.
            length_penalty (Optional[float], optional): Coefficient for exponential length penalty that is used with beam search. Only allowed for beam search. Defaults to 1.0. This is similar to Hugging Face's [`length_penalty`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.length_penalty) argument. Defaults to None.
            early_stopping (Optional[bool], optional): Whether to stop the beam search when at least `num_beams` beams are finished with the EOS token. Only allowed for beam search. Defaults to false. This is similar to Hugging Face's [`early_stopping`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.early_stopping) argument. Defaults to None.
            no_repeat_ngram (Optional[int], optional): If this exceeds 1, every ngram of that size can only occur once among the generated result (plus the input tokens for decoder-only models). 1 means that this mechanism is disabled (i.e., you cannot prevent 1-gram from being generated repeatedly). Defaults to 1. This is similar to Hugging Face's [`no_repeat_ngram_size`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.no_repeat_ngram_size) argument. Defaults to None.
            encoder_no_repeat_ngram (Optional[int], optional): If this exceeds 1, every ngram of that size occurring in the input token sequence cannot appear in the generated result. 1 means that this mechanism is disabled (i.e., you cannot prevent 1-gram from being generated repeatedly). Only allowed for encoder-decoder models. Defaults to 1. This is similar to Hugging Face's [`encoder_no_repeat_ngram_size`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.encoder_no_repeat_ngram_size) argument. Defaults to None.
            repetition_penalty (Optional[float], optional): Penalizes tokens that have already appeared in the generated result (plus the input tokens for decoder-only models). Should be greater than or equal to 1.0. 1.0 means no penalty. See [Keskar et al., 2019](https://arxiv.org/abs/1909.05858) for more details. This is similar to Hugging Face's [`repetition_penalty`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.repetition_penalty) argument. Defaults to None.
            encoder_repetition_penalty (Optional[float], optional): Penalizes tokens that have already appeaared in the input tokens. Should be greater than or equal to 1.0. 1.0 means no penalty. Only allowed for encoder-decoder models. See [Keskar et al., 2019](https://arxiv.org/abs/1909.05858) for more details. This is similar to Hugging Face's [`encoder_repetition_penalty`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.encoder_repetition_penalty) argument. Defaults to None.
            frequency_penalty (Optional[float], optional): Number between -2.0 and 2.0. Positive values penalizes tokens that have been sampled, taking into account their frequency in the preceding text. This penalization diminishes the model's tendency to reproduce identical lines verbatim. Defaults to None.
            presence_penalty (Optional[float], optional): Number between -2.0 and 2.0. Positive values penalizes tokens that have been sampled at least once in the existing text. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Non-zero positive numbers are allowed. Smaller temperature makes the generation result closer to greedy, argmax (i.e., `top_k = 1`) sampling. Defaults to 1.0. This is similar to Hugging Face's [`temperature`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.temperature) argument. Defaults to None.
            top_k (Optional[int], optional): The number of highest probability tokens to keep for sampling. Numbers between 0 and the vocab size of the model (both inclusive) are allowed. The default value is 0, which means that the API does not apply top-k filtering. This is similar to Hugging Face's [`top_k`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.top_k) argument. Defaults to None.
            top_p (Optional[float], optional): Tokens comprising the top `top_p` probability mass are kept for sampling. Numbers between 0.0 (exclusive) and 1.0 (inclusive) are allowed. Defaults to 1.0. This is similar to Hugging Face's [`top_p`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.top_p) argument. Defaults to None.
            stop (Optional[List[str]], optional): When one of the stop phrases appears in the generation result, the API will stop generation. The phrase is included in the generated result. If you are using beam search, all of the active beams should contain the stop phrase to terminate generation. Before checking whether a stop phrase is included in the result, the phrase is converted into tokens. We recommend using `stop_tokens` because it is clearer. For example, after tokenization, phrases "clear" and " clear" can result in different token sequences due to the prepended space character. Defaults to empty list. Defaults to None.
            stop_tokens (Optional[List[TokenSequenceParam]], optional): Same as the above `stop` field, but receives token sequences instead of text phrases. A TokenSequence type is a dict with the key 'tokens' and the value type List[int]. Defaults to None.
            seed (Optional[List[int]], optional): Seed to control random procedure. If nothing is given, the API generate the seed randomly, use it for sampling, and return the seed along with the generated result. When using the `n` argument, you can pass a list of seed values to control all of the independent generations. Defaults to None.
            token_index_to_replace (Optional[List[int]], optional): A list of token indices where to replace the embeddings of input tokens provided via either `tokens` or `prompt`. Defaults to None.
            embedding_to_replace (Optional[List[float]], optional): A list of flattened embedding vectors used for replacing the tokens at the specified indices provided via `token_index_to_replace`. Defaults to None.
            beam_search_type (Optional[BeamSearchType], optional): Which beam search type to use. `DETERMINISTIC` means the standard, deterministic beam search, which is similar to Hugging Face's [`beam_search`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search). Argmuents for controlling random sampling such as `top_k` and `top_p` are not allowed for this option. `STOCHASTIC` means stochastic beam search (more details in [Kool et al. (2019)](https://proceedings.mlr.press/v97/kool19a.html)). `NAIVE_SAMPLING` is similar to Hugging Face's [`beam_sample`](https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample). Defaults to `DETERMINISTIC`. Defaults to None.
            beam_compat_pre_normalization (Optional[bool], optional): Whether to perform normalization both before and after the perturbation. Defaults to None.
            beam_compat_no_post_normalization (Optional[bool], optional): When sets to `True`, do not perform normalization after the perturbation. Defaults to None.
            bad_words (Optional[List[str]], optional): Text phrases that should not be generated. For a bad word phrase that contains N tokens, if the first N-1 tokens appears at the last of the generated result, the logit for the last token of the phrase is set to -inf. We recommend using `bad_word_tokens` because it is clearer (more details in the document for `stop` field). Defaults to empty list. Defaults to None.
            bad_word_tokens (Optional[List[TokenSequenceParam]], optional): Same as the above `bad_words` field, but receives token sequences instead of text phrases. A TokenSequence type is a dict with the key 'tokens' and the value type List[int]. This is similar to Hugging Face's <a href="https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/text_generation#transformers.GenerationConfig.bad_words_ids(List[List[int]]," target="_top">`bad_word_ids`</a> argument. Defaults to None.
            include_output_logits (Optional[bool], optional): Whether to include the output logits to the generation output. Defaults to None.
            include_output_logprobs (Optional[bool], optional): Whether to include the output logprobs to the generation output. Defaults to None.
            forced_output_tokens (Optional[List[int]], optional): A token sequence that is enforced as a generation output. This option can be used when evaluating the model for the datasets with multi-choice problems (e.g., [HellaSwag](https://huggingface.co/datasets/hellaswag), [MMLU](https://huggingface.co/datasets/cais/mmlu)). Use this option with `include_output_logprobs` to get logprobs for the evaluation. Defaults to None.
            eos_token (Optional[List[int]], optional): A list of endpoint sentence tokens. Defaults to None.

        Raises:
            APIError: Raised when the HTTP API request to the deployment fails.
            SessionClosedError: Raised when the client session is not opened with `api_session()`.

        Returns:
            Union[AsyncCompletionStream, Completion]: If `stream` is `True`, a `AsyncCompletionStream` object that iterates the results per token is returned. Otherwise, a `Completion` object is returned.

        Examples:
            Basic usage:

            ```python
            import asyncio
            from friendli import Friendli

            client = Friendli(deployment_id="friendli-deployment-1b9483a0")

            async def main() -> None:
                completion = await client.completions.create(
                    prompt="Python is a popular language for",
                    stream=False,
                    max_tokens=100,
                    top_p=0.8,
                    temperature=0.5,
                    no_repeat_ngram=3,
                )
                print(completion.choices[0].text)

            asyncio.run(main())
            ```

            Usage of streaming mode:

            ```python
            import asyncio
            from friendli import Friendli

            client = Friendli(deployment_id="friendli-deployment-1b9483a0")

            async def main() -> None:
                stream = await client.completions.create(
                    prompt="Python is a popular language for",
                    stream=True,  # Enable stream mode.
                    max_tokens=100,
                    top_p=0.8,
                    temperature=0.5,
                    no_repeat_ngram=3,
                )
                async for line in stream:
                    print(line.choices[0].text, end="")

            asyncio.run(main())
            ```

        """
        request_dict = {
            "stream": stream,
            "prompt": prompt,
            "tokens": tokens,
            "timeout_microseconds": timeout_microseconds,
            "max_tokens": max_tokens,
            "max_total_tokens": max_total_tokens,
            "min_tokens": min_tokens,
            "min_total_tokens": min_total_tokens,
            "n": n,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
            "no_repeat_ngram": no_repeat_ngram,
            "encoder_no_repeat_ngram": encoder_no_repeat_ngram,
            "repetition_penalty": repetition_penalty,
            "encoder_repetition_penalty": encoder_repetition_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop": stop,
            "stop_tokens": stop_tokens,
            "seed": seed,
            "token_index_to_replace": token_index_to_replace,
            "embedding_to_replace": embedding_to_replace,
            "beam_search_type": beam_search_type,
            "beam_compat_pre_normalization": beam_compat_pre_normalization,
            "beam_compat_no_post_normalization": beam_compat_no_post_normalization,
            "bad_words": bad_words,
            "bad_word_tokens": bad_word_tokens,
            "include_output_logits": include_output_logits,
            "include_output_logprobs": include_output_logprobs,
            "forced_output_tokens": forced_output_tokens,
            "eos_token": eos_token,
        }
        response = await self._request(data=request_dict, stream=stream, model=model)

        if stream:
            return AsyncCompletionStream(response=response)
        return model_parse(Completion, response.json())


class CompletionStream(GenerationStream[CompletionLine]):
    """Completion stream."""

    def __next__(self) -> CompletionLine:  # noqa: D105
        line = next(self._iter)
        while not line:
            line = next(self._iter)

        parsed = json.loads(line.strip("data: "))
        try:
            return model_parse(CompletionLine, parsed)
        except ValidationError as exc:
            try:
                # The last iteration of the stream returns a response with `V1Completion` schema.
                model_parse(Completion, parsed)
                raise StopIteration from exc
            except ValidationError:
                raise InvalidGenerationError(
                    f"Generation result has invalid schema: {str(exc)}"
                ) from exc

    def wait(self) -> Optional[Completion]:
        """Waits for the generation to complete.

        Raises:
            InvalidGenerationError: Raised when the generation result has invalid format.

        Returns:
            Optional[Completion]: The full generation result.

        """
        for line in self._iter:
            if line:
                parsed = json.loads(line.strip("data: "))
                try:
                    # The last iteration of the stream returns a response with `V1Completion` schema.
                    return model_parse(Completion, parsed)
                except ValidationError as exc:
                    try:
                        # Skip the line response.
                        model_parse(CompletionLine, parsed)
                    except ValidationError:
                        raise InvalidGenerationError(
                            f"Generation result has invalid schema: {str(exc)}"
                        ) from exc
        return None


class AsyncCompletionStream(AsyncGenerationStream[CompletionLine]):
    """Asynchronous completion stream."""

    async def __anext__(self) -> CompletionLine:  # noqa: D105
        line = await self._iter.__anext__()
        while not line:
            line = await self._iter.__anext__()

        parsed = json.loads(line.strip("data: "))
        try:
            return model_parse(CompletionLine, parsed)
        except ValidationError as exc:
            try:
                # The last iteration of the stream returns a response with `V1Completion` schema.
                model_parse(Completion, parsed)
                raise StopAsyncIteration from exc
            except ValidationError:
                raise InvalidGenerationError(
                    f"Generation result has invalid schema: {str(exc)}"
                ) from exc

    async def wait(self) -> Optional[Completion]:  # noqa: D105
        """Waits for the generation to complete.

        Raises:
            InvalidGenerationError: Raised when the generation result has invalid format.

        Returns:
            Optional[Completion]: The full generation result.

        """
        async for line in self._iter:
            if line:
                parsed = json.loads(line.strip("data: "))
                try:
                    # The last iteration of the stream returns a response with `V1Completion` schema.
                    return model_parse(Completion, parsed)
                except ValidationError as exc:
                    try:
                        # Skip the line response.
                        model_parse(CompletionLine, parsed)
                    except ValidationError:
                        raise InvalidGenerationError(
                            f"Generation result has invalid schema: {str(exc)}"
                        ) from exc
        return None
