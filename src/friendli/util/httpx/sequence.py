# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Sequence generators."""

from __future__ import annotations

import random
from itertools import repeat
from typing import Iterator


def cap_sequence(seq: Iterator[float], *, cap: float) -> Iterator[float]:
    """Bound given sequence."""
    for val in seq:  # pragma: no branch
        yield min(val, cap)


def jitter_sequence(seq: Iterator[float]) -> Iterator[float]:
    """Jitter values by uniform sampling from zero to current value."""
    for val in seq:  # pragma: no branch
        yield random.uniform(0, val)  # noqa: S311


def constant_sequence(base: float) -> Iterator[float]:
    """Generates a sequence of constant values."""
    yield from repeat(base)


def exponential_sequence(base: float) -> Iterator[float]:
    """Generates a sequence of exponential values."""
    while True:
        yield base
        base *= 2


def fibonacci_sequence(base: float) -> Iterator[float]:
    """Generates a sequence of fibonacci values."""
    v = base

    while True:
        yield base
        base, v = v, base + v


def decorrelated_jitter_sequence(base: float, cap: float) -> Iterator[float]:
    """Generates a sequence of decorrelated jitter values."""
    value = base

    while True:
        value = min(cap, random.uniform(base, value * 3))
        yield value
