# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Email address validation callback."""

from __future__ import annotations

import re
from typing import Optional

import typer
from typer import Context

_EmailRegex = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")


def validate_email(value: Optional[str], ctx: Context) -> Optional[str]:
    """Validate an email address callback."""
    if ctx.resilient_parsing:
        return None

    if not value:
        return None

    if not _EmailRegex.fullmatch(value):
        msg = f'"{value}" is not a valid email address'
        raise typer.BadParameter(msg)

    return value
