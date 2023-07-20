# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CLI Interactive Prompt Utilities."""

from __future__ import annotations

import os
from subprocess import CalledProcessError, check_call
from typing import Optional

import typer


def open_editor(path: str, editor: Optional[str] = None):
    """Open the terminal editor."""
    default_editor = editor or get_default_editor()
    try:
        check_call([default_editor, path])
    except CalledProcessError:
        typer.secho("", fg=typer.colors.RED)


def get_default_editor() -> str:
    """Get default terminal editor."""
    return os.environ.get("PERIFLOW_CLI_EDITOR", "vim")
