# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Auto-generate CLI docs."""


from __future__ import annotations

import json
import os
from typing import List, Optional, Union, cast

import typer
from click import Choice
from typer.core import TyperArgument, TyperCommand, TyperOption

from periflow.cli import app


def _generate_category_json(
    docs_root: str, groups: List[str], force: bool = False
) -> None:
    path = os.path.join(docs_root, *groups, "_category_.json")
    content = {
        "label": f"pf {' '.join(groups)}",
        "collapsible": True,
        "collapsed": False,
    }

    if force:
        os.remove(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=4)


# pylint: disable=too-many-arguments
def _generate_command_mdx(
    docs_root: str,
    groups: Optional[List[str]],
    command_name: str,
    usage: str,
    summary: Optional[str],
    arguments: str,
    options: str,
    force: bool = False,
) -> None:
    if groups is None:
        header = f"# pf {command_name}"
        path = os.path.join(docs_root, f"{command_name}.mdx")
    else:
        header = f"# pf {' '.join(groups)} {command_name}"
        path = os.path.join(docs_root, *groups, f"{command_name}.mdx")

    contents = f"""
{header}

## Usage

```bash
{usage}
```

## Summary

{summary}
""".lstrip()

    if arguments:
        contents += f"""
## Arguments

| Argument | Type | Summary | Default | Required | 
|----------|------|---------|---------|----------|
{arguments.rstrip()}
"""

    if options:
        contents += f"""
## Options

| Option | Type | Summary | Default | Required |
|--------|------|---------|---------|----------|
{options.rstrip()}
"""

    if force:
        os.remove(path)
    with open(path, "w", encoding="utf-8") as file:
        file.write(contents)


def generate_app_docs(
    docs_root: str,
    app: typer.Typer,
    groups: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """Generate CLI app documentation."""
    # Create sub-command group folder.
    if groups is not None:
        group_dir_name = os.path.join(docs_root, *groups)
        os.makedirs(group_dir_name, exist_ok=force)
        _generate_category_json(docs_root=docs_root, groups=groups)

    for command_info in app.registered_commands:
        command = cast(
            TyperCommand,
            typer.main.get_command_from_info(
                command_info=command_info,
                pretty_exceptions_short=True,
                rich_markup_mode=None,
            ),
        )
        prefix = f"pf {' '.join(groups)}" if groups else "pf"
        usage_str = f"{prefix} {command.name}"
        usage_args_str = ""
        option_strs = []
        arg_strs = []
        options_exist = False
        for param in command.params:
            param = cast(Union[TyperOption, TyperArgument], param)

            if isinstance(param.type, Choice):
                type_val = f"CHOICE: [{', '.join(param.type.choices)}]"
            else:
                type_val = param.type.name.upper()

            opts = [f"`{opt}`" for opt in param.opts]
            if param.required:
                opts = [f"**{opt}**" for opt in opts]

            opt_spec = f"| {', '.join(opts)} | {type_val} | {param.help} | {'-' if param.required else str(param.default)} | {'✅' if param.required else '❌'} |"  # pylint: disable=line-too-long
            if param.param_type_name == "option":
                options_exist = True
                option_strs.append(opt_spec)
            if param.param_type_name == "argument":
                usage_args_str += f" {cast(str, param.name).upper()}"
                arg_strs.append(opt_spec)

        if options_exist:
            usage_str += f" {command.options_metavar}"
        usage_str += usage_args_str

        assert command.name is not None
        _generate_command_mdx(
            docs_root=docs_root,
            groups=groups,
            command_name=command.name,
            usage=usage_str,
            summary=command.help,
            arguments="\n".join(arg_strs),
            options="\n".join(option_strs),
        )


def generate_group_docs(
    docs_root: str,
    app: typer.Typer,
    force: bool,
    parent_groups: Optional[List[str]] = None,
) -> None:
    """Generate documentation for nested commands."""
    for group in app.registered_groups:
        subapp = group.typer_instance
        if subapp is not None:
            if subapp.registered_groups:
                # Recursively generate multi-nested command docs.
                generate_group_docs(
                    docs_root=docs_root,
                    app=subapp,
                    force=force,
                    parent_groups=parent_groups or [] + [group.name],
                )

            assert group.name is not None
            if parent_groups:
                groups = [*parent_groups, group.name]
            else:
                groups = [group.name]
            generate_app_docs(
                docs_root=docs_root, app=subapp, groups=groups, force=force
            )


def autodoc(
    output_path: str = typer.Option(
        ..., "--output-path", "-o", help="Path to save generated docs."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing documents."
    ),
) -> None:
    """Auto-generate documentation."""
    if not os.path.isdir(output_path):
        typer.secho(f"Save direcotry '{output_path}' does not exist.")
        raise typer.Abort()

    generate_app_docs(docs_root=output_path, app=app, force=force)
    generate_group_docs(docs_root=output_path, app=app, force=force)


if __name__ == "__main__":
    typer.run(autodoc)
