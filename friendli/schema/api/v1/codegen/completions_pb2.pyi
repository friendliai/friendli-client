# Copyright (c) 2023-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class V1CompletionsRequest(_message.Message):
    __slots__ = [
        "stream",
        "prompt",
        "tokens",
        "timeout_microseconds",
        "max_tokens",
        "max_total_tokens",
        "min_tokens",
        "min_total_tokens",
        "n",
        "num_beams",
        "length_penalty",
        "early_stopping",
        "no_repeat_ngram",
        "encoder_no_repeat_ngram",
        "repetition_penalty",
        "encoder_repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "temperature",
        "top_k",
        "top_p",
        "stop",
        "stop_tokens",
        "seed",
        "token_index_to_replace",
        "embedding_to_replace",
        "beam_search_type",
        "beam_compat_pre_normalization",
        "beam_compat_no_post_normalization",
        "bad_words",
        "bad_word_tokens",
        "include_output_logits",
        "include_output_logprobs",
        "forced_output_tokens",
        "eos_token",
    ]

    class BeamSearchType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DETERMINISTIC: _ClassVar[V1CompletionsRequest.BeamSearchType]
        STOCHASTIC: _ClassVar[V1CompletionsRequest.BeamSearchType]
        NAIVE_SAMPLING: _ClassVar[V1CompletionsRequest.BeamSearchType]
    DETERMINISTIC: V1CompletionsRequest.BeamSearchType
    STOCHASTIC: V1CompletionsRequest.BeamSearchType
    NAIVE_SAMPLING: V1CompletionsRequest.BeamSearchType

    class TokenSequence(_message.Message):
        __slots__ = ["tokens"]
        TOKENS_FIELD_NUMBER: _ClassVar[int]
        tokens: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, tokens: _Optional[_Iterable[int]] = ...) -> None: ...
    STREAM_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MIN_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MIN_TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    NUM_BEAMS_FIELD_NUMBER: _ClassVar[int]
    LENGTH_PENALTY_FIELD_NUMBER: _ClassVar[int]
    EARLY_STOPPING_FIELD_NUMBER: _ClassVar[int]
    NO_REPEAT_NGRAM_FIELD_NUMBER: _ClassVar[int]
    ENCODER_NO_REPEAT_NGRAM_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    ENCODER_REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STOP_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    TOKEN_INDEX_TO_REPLACE_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_TO_REPLACE_FIELD_NUMBER: _ClassVar[int]
    BEAM_SEARCH_TYPE_FIELD_NUMBER: _ClassVar[int]
    BEAM_COMPAT_PRE_NORMALIZATION_FIELD_NUMBER: _ClassVar[int]
    BEAM_COMPAT_NO_POST_NORMALIZATION_FIELD_NUMBER: _ClassVar[int]
    BAD_WORDS_FIELD_NUMBER: _ClassVar[int]
    BAD_WORD_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_OUTPUT_LOGITS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_OUTPUT_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    FORCED_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_FIELD_NUMBER: _ClassVar[int]
    stream: bool
    prompt: str
    tokens: _containers.RepeatedScalarFieldContainer[int]
    timeout_microseconds: int
    max_tokens: int
    max_total_tokens: int
    min_tokens: int
    min_total_tokens: int
    n: int
    num_beams: int
    length_penalty: float
    early_stopping: bool
    no_repeat_ngram: int
    encoder_no_repeat_ngram: int
    repetition_penalty: float
    encoder_repetition_penalty: float
    frequency_penalty: float
    presence_penalty: float
    temperature: float
    top_k: int
    top_p: float
    stop: _containers.RepeatedScalarFieldContainer[str]
    stop_tokens: _containers.RepeatedCompositeFieldContainer[
        V1CompletionsRequest.TokenSequence
    ]
    seed: _containers.RepeatedScalarFieldContainer[int]
    token_index_to_replace: _containers.RepeatedScalarFieldContainer[int]
    embedding_to_replace: _containers.RepeatedScalarFieldContainer[float]
    beam_search_type: V1CompletionsRequest.BeamSearchType
    beam_compat_pre_normalization: bool
    beam_compat_no_post_normalization: bool
    bad_words: _containers.RepeatedScalarFieldContainer[str]
    bad_word_tokens: _containers.RepeatedCompositeFieldContainer[
        V1CompletionsRequest.TokenSequence
    ]
    include_output_logits: bool
    include_output_logprobs: bool
    forced_output_tokens: _containers.RepeatedScalarFieldContainer[int]
    eos_token: _containers.RepeatedScalarFieldContainer[int]
    def __init__(
        self,
        stream: bool = ...,
        prompt: _Optional[str] = ...,
        tokens: _Optional[_Iterable[int]] = ...,
        timeout_microseconds: _Optional[int] = ...,
        max_tokens: _Optional[int] = ...,
        max_total_tokens: _Optional[int] = ...,
        min_tokens: _Optional[int] = ...,
        min_total_tokens: _Optional[int] = ...,
        n: _Optional[int] = ...,
        num_beams: _Optional[int] = ...,
        length_penalty: _Optional[float] = ...,
        early_stopping: bool = ...,
        no_repeat_ngram: _Optional[int] = ...,
        encoder_no_repeat_ngram: _Optional[int] = ...,
        repetition_penalty: _Optional[float] = ...,
        encoder_repetition_penalty: _Optional[float] = ...,
        frequency_penalty: _Optional[float] = ...,
        presence_penalty: _Optional[float] = ...,
        temperature: _Optional[float] = ...,
        top_k: _Optional[int] = ...,
        top_p: _Optional[float] = ...,
        stop: _Optional[_Iterable[str]] = ...,
        stop_tokens: _Optional[
            _Iterable[_Union[V1CompletionsRequest.TokenSequence, _Mapping]]
        ] = ...,
        seed: _Optional[_Iterable[int]] = ...,
        token_index_to_replace: _Optional[_Iterable[int]] = ...,
        embedding_to_replace: _Optional[_Iterable[float]] = ...,
        beam_search_type: _Optional[
            _Union[V1CompletionsRequest.BeamSearchType, str]
        ] = ...,
        beam_compat_pre_normalization: bool = ...,
        beam_compat_no_post_normalization: bool = ...,
        bad_words: _Optional[_Iterable[str]] = ...,
        bad_word_tokens: _Optional[
            _Iterable[_Union[V1CompletionsRequest.TokenSequence, _Mapping]]
        ] = ...,
        include_output_logits: bool = ...,
        include_output_logprobs: bool = ...,
        forced_output_tokens: _Optional[_Iterable[int]] = ...,
        eos_token: _Optional[_Iterable[int]] = ...,
    ) -> None: ...
