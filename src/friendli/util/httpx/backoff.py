# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Sequence generators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generator, Iterator, Literal

from .sequence import (
    cap_sequence,
    constant_sequence,
    decorrelated_jitter_sequence,
    exponential_sequence,
    fibonacci_sequence,
    jitter_sequence,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

SequenceType: TypeAlias = Literal[
    "constant", "exponential", "fibonacci", "decorrelated_jitter"
]


LookUpTable: dict[SequenceType, Callable[[float], Iterator[float]]] = {
    "constant": constant_sequence,
    "exponential": exponential_sequence,
    "fibonacci": fibonacci_sequence,
}


def generate_sequence(
    seq_type: SequenceType,
    *,
    base: float = 1,
    jitter: bool = False,
    cap: float | None = None,
) -> Generator[float, None, None]:
    """Generates a sequence.

    Args:
        seq_type (SequenceType): sequence type
        jitter (bool): if true, jitter values. Defaults to False.
        base (float): sequence base value. Defaults to zero.
        cap (float | None): sequence values cap bound. Defaults to None.

    """
    if base <= 0:
        msg = "base must be positive"
        raise ValueError(msg)

    if cap is not None and cap < base:
        msg = "cap must be greater than base"
        raise ValueError(msg)

    try:
        sequence = LookUpTable[seq_type](base)
    except KeyError:
        raise NotImplementedError(seq_type) from None

    if jitter:
        sequence = jitter_sequence(sequence)
    if cap is not None:
        sequence = cap_sequence(sequence, cap=cap)

    yield from sequence


def generate_decorrelated_jitter_sequence(
    *,
    base: float = 0,
    cap: float,
) -> Generator[float, None, None]:
    """Generates a sequence of decorrelated jitter values.

    Args:
        base (float): sequence base value. Defaults to zero.
        cap (float | None): sequence values cap bound. Defaults to None.

    """
    if base < 0:
        msg = "base must be non-negative"
        raise ValueError(msg)

    if cap < base:
        msg = "cap must be greater than base"
        raise ValueError(msg)

    sequence = decorrelated_jitter_sequence(base, cap)
    yield from sequence
