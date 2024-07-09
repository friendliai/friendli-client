# Copyright (c) 2021-present, FriendliAI Inc. All rights reserved.

"""Typer utilities."""


from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommandUsageExample:
    """A command example.

    Attributes:
         synopsis (str): A short description of the example.
         args (str): The command line arguments to run.

    """

    synopsis: str
    args: str


def format_examples(examples: list[CommandUsageExample]) -> str:
    """Format list of examples into a help string."""
    # Note: typer.rich_utils:rich_format_help changes double new lines to single
    #       new lines, so we need to add 4 new lines to get 2 new lines.
    #       Also, indentation is removed. so we use blank emojis to indent.
    lines = ["[dim]Examples:[/dim]"]

    for example in examples:
        lines.append(f"- {example.synopsis}")
        lines.append(f"ㅤ[cyan]$ {example.args}[/cyan]")

    return "\n\n\n\n".join(lines)
