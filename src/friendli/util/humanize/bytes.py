# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Large numbers from/to human friendly format.

This is non-si format.
"""

from __future__ import annotations

from math import floor, log2
from re import IGNORECASE, compile
from typing import Final

# IEC Sizes.
Byte: Final = 1
KiByte: Final = Byte * 1024
MiByte: Final = KiByte * 1024
GiByte: Final = MiByte * 1024
TiByte: Final = GiByte * 1024
PiByte: Final = TiByte * 1024
EiByte: Final = PiByte * 1024

IByte: Final = Byte
KByte: Final = IByte * 1000
MByte: Final = KByte * 1000
GByte: Final = MByte * 1000
TByte: Final = GByte * 1000
PByte: Final = TByte * 1000
EByte: Final = PByte * 1000

_ByteStringRe: Final = compile(r"^([\d.]+)\s?([a-z]?i?b?)$", IGNORECASE)
_ByteSuffixLookup: Final = {
    "b": Byte,
    "kib": KiByte,
    "kb": KByte,
    "mib": MiByte,
    "mb": MByte,
    "gib": GiByte,
    "gb": GByte,
    "tib": TiByte,
    "tb": TByte,
    "pib": PiByte,
    "pb": PByte,
    "eib": EiByte,
    "eb": EByte,
    # Without suffix
    "ki": KiByte,
    "k": KByte,
    "mi": MiByte,
    "m": MByte,
    "gi": GiByte,
    "g": GByte,
    "ti": TiByte,
    "t": TByte,
    "pi": PiByte,
    "p": PByte,
    "ei": EiByte,
    "e": EByte,
}

_SiUnits = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
_IecUnits = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]


def parse_bytes(_s: str) -> float:
    """Parse a human-readable byte string into a number value.

    Args:
        _s: the string to parse.

    Returns:
        parsed number.

    Raises:
        ValueError: if the string cannot be parsed.

    """
    s = _s.replace(",", "")
    matches = _ByteStringRe.fullmatch(s)

    if matches is None:
        msg = f"Cannot parse bytes {s}"
        raise ValueError(msg)

    digit = float(matches[1])
    suffix = matches[2]

    if suffix:
        suffix_num = _ByteSuffixLookup[suffix.lower()]
        digit *= suffix_num

    return digit


def format_bytes(v: float) -> str:
    """Format a number to a human readable byte string.

    Args:
        v: the number to be formatted.

    Returns:
        the formatted string.

    """
    return _humanize_bytes(v, 1000, _SiUnits)


def format_ibytes(v: float) -> str:
    """Format a number to a human readable byte string.

    Args:
        v: the number to be formatted.

    Returns:
        the formatted string.

    """
    return _humanize_bytes(v, 1024, _IecUnits)


def _humanize_bytes(v: float, base: int, units: list[str]) -> str:
    """Format a number to a human readable byte string.

    Args:
        v: the number to be formatted.
        base: the base of the number.
        units: the list of units.

    Returns:
        the formatted string.

    """
    if v < 10:  # noqa: PLR2004
        return f"{v} {units[0]}"

    exp = floor(log2(v) / log2(base))
    unit = units[exp]

    val = round((v / base**exp) * 10) / 10

    if val < 10:  # noqa: PLR2004
        return f"{val:.1f} {unit}"

    return f"{int(val)} {unit}"
