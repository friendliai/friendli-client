# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Script to attach copyright header."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List

import typer


def format_copyright_header(
    paths: List[str] = typer.Argument(..., help="List of paths to evaluate."),
    check_only: bool = typer.Option(False, "--check", help="Check copyright headers."),
) -> None:
    """Check for copyright headers."""
    # get copyright header configuration
    current_year = datetime.now().year
    header = (
        f"# Copyright (c) {current_year}-present, FriendliAI Inc. All rights reserved."
    )
    header_pattern = (
        r"# Copyright \(c\) \b\d{4}\b-present, FriendliAI Inc. All rights reserved."
    )

    file_exts = ["**/*.py", "**/*.pyi"]
    for path in paths:
        for pattern in file_exts:
            for file in Path(path).rglob(pattern):
                with open(file, "r", encoding="utf-8") as prev_file:
                    content = prev_file.read()
                    prev_file.seek(0)
                    first_line_content = prev_file.readline()

                if re.match(header_pattern, first_line_content):
                    continue

                if not first_line_content:  # empty file
                    first_line_content = header + "\n"
                elif check_only:
                    typer.secho(
                        f"Invalid copyright header found in {file}!",
                        fg=typer.colors.RED,
                    )
                else:
                    new_content = header + "\n\n" + content

                with open(file, "w", encoding="utf-8") as new_file:
                    new_file.write(new_content)


if __name__ == "__main__":
    typer.run(format_copyright_header)
