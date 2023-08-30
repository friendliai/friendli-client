# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Auto-generate SDK docs."""

from __future__ import annotations

import ast
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import typer
from docstring_parser import (
    Docstring,
    DocstringParam,
    DocstringRaises,
    DocstringReturns,
    DocstringStyle,
    parse,
)
from docstring_parser.common import DocstringExample

SKIP_DOC = "[skip-doc]"


class NodeDef(ABC):
    """Node definition base interface."""

    @abstractmethod
    def stringify(self) -> str:
        """Convert the node info to string."""


@dataclass
class ParamDef(NodeDef):
    """Parameter definition."""

    name: str
    type: Optional[str] = None
    default: Optional[str] = None

    def stringify(self) -> str:
        """Convert the parameter info to string."""
        res = f"{self.name}: {self.type}"
        if self.default:
            res += f" = {self.default}"
        return res


@dataclass
class FuncDef(NodeDef):
    """Function definition."""

    name: str
    params: List[ParamDef]
    returns: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False

    def stringify(self) -> str:
        """Convert the function info to string."""
        func_type = "async def" if self.is_async else "def"

        delimiter = ",\n    "
        if self.params:
            param_signature = (
                f"\n    {delimiter.join(param.stringify() for param in self.params)}\n"
            )
        else:
            param_signature = ""
        res = f"""
### _{func_type}_ `{self.name}`

```python
{self.name}({param_signature}) -> {self.returns}
```
""".lstrip()

        if self.docstring:
            res += generate_docstring_mdx(
                parse(self.docstring, style=DocstringStyle.GOOGLE)
            )

        return res


@dataclass
class ClassDef(NodeDef):
    """Class definition."""

    name: str
    bases: List[str]
    attributes: List[ParamDef]
    methods: List[FuncDef]
    docstring: Optional[str] = None

    def stringify(self) -> str:
        """Convert the class info to string."""
        res = f"""
# _class_ `{self.name}({', '.join(attr.stringify() for attr in self.attributes)})`

""".lstrip()

        if self.bases:
            res += f"""
> Bases: {', '.join(self.bases)}

""".lstrip()

        res += self.docstring or ""
        res += "\n"

        if self.methods:
            res += "## Methods\n\n"

        res += "\n".join(method.stringify() for method in self.methods)
        return res


def generate_params_mdx(params: List[DocstringParam]) -> str:
    """Generate MDX format string of parameter docs."""
    res = """
#### Arguments

| Name | Type | Description | Default | Optional |
|------|------|-------------|---------|----------|
"""

    for param in params:
        if param.default:
            default_value = f"`{param.default}`"
        else:
            default_value = "-"
        res += f"""
| `{param.arg_name}` | `{param.type_name}` | {param.description} | {default_value} | {"✅" if param.is_optional else "❌"} |
""".lstrip()

    return res


def generate_returns_mdx(returns: DocstringReturns) -> str:
    """Generate MDX format string of returns docs."""
    if returns.is_generator:
        header = "Yields"
    else:
        header = "Returns"

    return f"""
#### {header}

| Type | Description |
|------|-------------|
| `{returns.type_name}` | {returns.description} |
"""


def generate_raises_mdx(raises: List[DocstringRaises]) -> str:
    """Generate MDX format string of raises docs."""
    res = """
#### Raises

| Type | Description |
|------|-------------|
""".lstrip()

    for r in raises:
        res += f"""
| `{r.type_name}` | {r.description} |
""".lstrip()

    return res


def generate_examples_mdx(example: DocstringExample) -> str:
    """Generate MDX format string of examples docs."""
    return f"""
#### Examples

{example.description}
"""


def generate_docstring_mdx(docstring: Docstring) -> str:
    """Generate MDX format string of docstring docs."""
    res = f"""
{docstring.short_description}
"""

    # Create argument docs
    if docstring.params:
        res += generate_params_mdx(docstring.params)

    if docstring.returns:
        res += generate_returns_mdx(docstring.returns)

    if docstring.raises:
        res += generate_raises_mdx(docstring.raises)

    if docstring.long_description:
        res += f"{docstring.long_description}\n"

    if docstring.examples:
        for example in docstring.examples:
            res += generate_examples_mdx(example)

    return res


def check_skip(
    node: Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]
) -> bool:
    """Check if the node is marked as a skip target."""
    docstring = ast.get_docstring(node)
    return docstring is not None and SKIP_DOC in docstring


def get_node_repr(node: ast.AST) -> str:
    """Get a string representation of the AST node."""
    # TODO: Add more cases.
    if isinstance(node, ast.Name):
        res = node.id
    elif isinstance(node, ast.Subscript):
        res = f"{get_node_repr(node.value)}[{get_node_repr(node.slice)}]"
    elif isinstance(node, ast.Tuple):
        res = f"Tuple[{', '.join(str(get_node_repr(e)) for e in node.elts)}]"
    elif isinstance(node, ast.Attribute):
        res = f"{get_node_repr(node.value)}.{node.attr}"
    elif isinstance(node, ast.Call):
        res = f"{get_node_repr(node.func)}(...)"
    elif isinstance(node, ast.Constant):
        res = node.value
    res = ast.unparse(node)

    return res


def extract_class_attrs(node: ast.ClassDef) -> Optional[List[ParamDef]]:
    """Extract class attributes."""
    init_fn = None
    for method in node.body:
        if isinstance(method, ast.FunctionDef) and method.name == "__init__":
            init_fn = extract_function_info(method, skip_params=["self", "cls"])
            return init_fn.params
    return None


def extract_function_info(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    skip_params: Optional[List[str]] = None,
) -> FuncDef:
    """Extract function info."""
    params = []
    for arg, default in zip(
        node.args.args, _pad_none_left(node.args.defaults, len(node.args.args))
    ):
        if skip_params and arg.arg in skip_params:
            continue

        params.append(
            ParamDef(
                name=arg.arg,  # type: ignore
                type=arg.annotation and get_node_repr(arg.annotation),  # type: ignore
                default=default and str(get_node_repr(default)),  # type: ignore
            ),
        )

    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if skip_params and arg.arg in skip_params:
            continue

        params.append(
            ParamDef(
                name=arg.arg,  # type: ignore
                type=arg.annotation and get_node_repr(arg.annotation),  # type: ignore
                default=default and str(get_node_repr(default)),
            ),
        )

    return FuncDef(
        name=node.name,
        params=params,
        returns=node.returns and get_node_repr(node.returns),  # type: ignore
        docstring=ast.get_docstring(node),
        is_async=isinstance(node, ast.AsyncFunctionDef),
    )


def extract_defs(
    file_path: str,
) -> List[Union[ClassDef, FuncDef]]:
    """Extract all info of functions and classes in the file."""
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read())

    defs: List[Union[ClassDef, FuncDef]] = []

    processed_lineno = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if check_skip(node):
                continue

            # Find method nodes.
            methods = []
            for method in node.body:
                if (
                    isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name != "__init__"
                    and not method.name.startswith("_")
                ):
                    if check_skip(method):
                        continue

                    method_def = extract_function_info(
                        method, skip_params=["self", "cls"]
                    )
                    if method_def is not None:
                        methods.append(method_def)
                        processed_lineno.append(method.lineno)

            class_info = ClassDef(
                name=node.name,
                bases=[get_node_repr(n) for n in node.bases],
                attributes=extract_class_attrs(node) or [],
                docstring=ast.get_docstring(node),
                methods=methods,
            )

            defs.append(class_info)
        elif (
            isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")
            and node.lineno not in processed_lineno
        ):
            if check_skip(node):
                continue

            function_info = extract_function_info(node)
            defs.append(function_info)

    return defs


def _pad_none_left(l: List[Any], desired_size: int) -> List[Any]:
    list_size = len(l)
    assert desired_size >= list_size
    pad_count = desired_size - list_size
    return [None] * pad_count + l


def autodoc(
    source_paths: List[str] = typer.Argument(
        ..., help="Paths to the source code file to documentize."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", "-o", help="Path to save generated docs."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing documents."
    ),
):
    """Auto-generate documentation."""
    for path in source_paths:
        node_defs = extract_defs(path)

        res = f"""
# {os.path.basename(path).split(".", maxsplit=1)[0].capitalize()}

<a href="https://github.com/friendliai/periflow-client/tree/main/{path}">
    <img src="https://img.shields.io/badge/View%20source%20on%20GitHub-181717?style=for-the-badge&logo=Github" />
</a>

""".lstrip()

        res += "\n".join(node_def.stringify() for node_def in node_defs)

        save_path = os.path.join(
            output_path, f"{os.path.basename(path).split('.', maxsplit=1)[0]}.mdx"
        )

        if os.path.isfile(save_path) and force:
            os.remove(save_path)

        with open(save_path, "w", encoding="utf-8") as file:
            file.write(res)

        typer.secho(
            f"Docs for {path} is generated successfully at {save_path}.",
            fg=typer.colors.GREEN,
        )


if __name__ == "__main__":
    typer.run(autodoc)
