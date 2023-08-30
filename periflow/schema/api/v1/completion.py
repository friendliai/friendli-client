# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow V1 Completion Serving API Schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from periflow.enums import BeamSearchType


class TokenSequence(BaseModel):
    """Token sequence schema."""

    tokens: List[int]


class V1CompletionOptions(BaseModel):
    """V1 completion options schema."""

    stream: Optional[bool] = None  # Enable streaming mode.
    prompt: Optional[str] = None  # Input prompt.
    tokens: Optional[List[int]] = None  # Input token sequence.
    timeout_microseconds: Optional[int] = None  # Request timeout in ms.
    max_tokens: Optional[int] = None  # Maximum output token count.
    max_total_tokens: Optional[int] = None  # Maximum input + output token count.
    min_tokens: Optional[int] = None  # Minimum output token count.
    min_total_tokens: Optional[int] = None  # Minimum input + output token count.
    n: Optional[int] = None  # The number of output sequences.
    num_beams: Optional[int] = None  # The number of beams in the beam search.
    length_penalty: Optional[float] = None  # Length penalty in the beam search.
    early_stopping: Optional[bool] = None  # Enable early stopping.
    no_repeat_ngram: Optional[int] = None  # Make N-gram not appear from the output.
    encoder_no_repeat_ngram: Optional[
        int
    ] = None  # Make N-gram not appear from the encoder output.
    repetition_penalty: Optional[float] = None  # Repetition penalty.
    encoder_repetition_penalty: Optional[float] = None  # Encoder repetition penality.
    temperature: Optional[float] = None  # Temperature.
    top_k: Optional[int] = None  # Top K.
    top_p: Optional[float] = None  # Top P.
    stop: Optional[List[str]] = None  # List of stop words.
    stop_tokens: Optional[List[TokenSequence]] = None  # List of stop tokens.
    seed: Optional[List[int]] = None  # Seed.
    beam_search_type: Optional[BeamSearchType] = None  # Beam search type.
    beam_compat_pre_normalization: Optional[bool] = None
    beam_compat_no_post_normalization: Optional[bool] = None
    bad_words: Optional[List[str]] = None  # List of bad words.
    bad_word_tokens: Optional[List[TokenSequence]] = None  # List of bad word tokens.
    include_output_logits: Optional[bool] = None  # Include logits in the output.
    include_output_logprobs: Optional[bool] = None  # Include logprobs in the output.
    forced_output_tokens: Optional[
        List[int]
    ] = None  # List of tokens enforced to be generated.
    eos_token: Optional[List[int]] = None  # List of EOS tokens.


class V1CompletionChoice(BaseModel):
    """V1 completion choice schema."""

    index: int
    seed: int
    text: str
    tokens: List[int]


class V1Completion(BaseModel):
    """V1 completion schema."""

    choices: List[V1CompletionChoice]


class V1CompletionLine(BaseModel):
    """V1 completion line schema."""

    event: str
    index: int
    text: str
    token: int
