# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Script to attach license header."""

from __future__ import annotations

from pathlib import Path
from typing import List

import toml
import typer


def format_license_header(
    paths: List[str] = typer.Argument(..., help="List of paths to evaluate."),
    config_file: str = typer.Option(
        ..., "-c", "--config", help="Path to the config file."
    ),
    check_only: bool = typer.Option(False, "--check", help="Check license headers."),
) -> None:
    """Check for license headers."""
    # get license header configuration
    with open(config_file, "r", encoding="utf-8") as cfg_file:
        pyproject_config = toml.load(cfg_file)

    if "license" not in pyproject_config["tool"]:
        typer.secho("No license header provided found.", fg=typer.colors.RED)
        return

    if "header" not in pyproject_config["tool"]["license"]:
        typer.secho(
            "No header value provided in license configuration.", fg=typer.colors.RED
        )

    header = "# " + pyproject_config["tool"]["license"]["header"]
    space = pyproject_config["tool"]["license"].get("space", 1)

    header_with_space = header + (space + 1) * "\n"

    for path in paths:
        for file in Path(path).rglob("**/*.py"):
            with open(file, "r", encoding="utf-8") as prev_file:
                content = prev_file.read()

            if content.startswith(header_with_space):
                continue

            content = content.replace(header, "").strip()
            assert header not in content

            if not content:  # empty file
                content = header + "\n"
            elif check_only:
                typer.secho(
                    f"Invalid license header found in {file}!", fg=typer.colors.RED
                )
            else:
                content = header_with_space + content

            with open(file, "w", encoding="utf-8") as new_file:
                new_file.write(content)


if __name__ == "__main__":
    typer.run(format_license_header)
