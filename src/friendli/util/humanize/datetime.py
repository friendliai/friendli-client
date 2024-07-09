# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Humanize datetime."""

from __future__ import annotations

from datetime import datetime


def humanize_datetime(  # noqa: C901, PLR0911
    date: datetime,
    from_: datetime | None = None,
    *,
    shorthand: bool = False,
) -> str:
    """Humanize datetime into relative time.

    Args:
        date: the datetime to be humanized.
        from_: the reference datetime.
        shorthand: whether to use shorthand.

    Returns:
        The humanized datetime.

    """
    from_ = from_ or datetime.now(tz=date.tzinfo)

    if date >= from_:  # pragma: no cover
        msg = "method only supports formatting dates in the past."
        raise NotImplementedError(msg)

    delta = from_ - date
    seconds = int(delta.total_seconds())

    if seconds <= 1:
        return "a second ago" if not shorthand else "1s ago"

    if seconds < 59:  # noqa: PLR2004
        return f"{seconds} seconds ago" if not shorthand else f"{seconds}s ago"

    minutes = seconds // 60
    if minutes <= 1:
        return "a minute ago" if not shorthand else "1m ago"

    if minutes < 59:  # noqa: PLR2004
        return f"{minutes} minutes ago" if not shorthand else f"{minutes}m ago"

    hours = minutes // 60
    if hours <= 1:
        return "an hour ago" if not shorthand else "1h ago"

    if hours < 23:  # noqa: PLR2004
        return f"{hours} hours ago" if not shorthand else f"{hours}h ago"

    days = hours // 24
    if days <= 1:
        return "a day ago" if not shorthand else "1d ago"

    if days < 7:  # noqa: PLR2004
        return f"{days} days ago" if not shorthand else f"{days}d ago"

    weeks = days // 7
    if weeks <= 1:
        return "a week ago" if not shorthand else "1w ago"

    if weeks < 52:  # noqa: PLR2004
        return f"{weeks} weeks ago" if not shorthand else f"{weeks}w ago"

    years = weeks // 52
    if years <= 1:
        return "a year ago" if not shorthand else "1y ago"

    return f"{years} years ago" if not shorthand else f"{years}y ago"
