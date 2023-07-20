# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""PeriFlow CLI Output Formatter."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

from rich import box
from rich.console import Console, RenderableType
from rich.filesize import decimal
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


def get_value(data: Dict[str, Any], keys: str) -> str:
    """Get value of `key` from `data`.

    Unlike dict.get method, it is available to access the nested value in `data`.

    Args:
        data (Dict[str, Any]): Data to read.
        key (str): Dot(.)-separated nested keys in data to access the value.

    Returns:
        T: The retrieved data.

    """
    value: Any = data
    key_list = keys.split(".")
    for key in key_list:
        # e.g. `data.results[2].a`
        getitem_match = re.match(r"(.+)\[(-?\d+)\]$", key)
        if getitem_match:
            value = value.get(getitem_match.group(1))[int(getitem_match.group(2))]
        else:
            value = value.get(key)
    if isinstance(value, Enum):
        value = value.value
    if value is None:
        return "-"
    return str(value)


@dataclass
class Formatter:
    """Formatter helps data formatting and visualization."""

    name: str

    def __post_init__(self) -> None:
        """Post-init formatter."""
        self._console = Console()


@dataclass
class ListFormatter(Formatter):
    """Base interface for list data formatter."""

    fields: List[str]
    headers: List[str]
    extra_fields: List[str] = field(default_factory=list)
    extra_headers: List[str] = field(default_factory=list)
    substitute_exact_match_only: bool = True

    def __post_init__(self) -> None:
        """Post-init formatter."""
        super().__post_init__()
        assert len(self.fields) == len(self.headers)
        assert len(self.extra_fields) == len(self.extra_headers)

        self._styling_map: Dict[str, Any] = {}
        self._substitution_rule: Dict[str, Any] = {}

    def render(self, data: List[Dict[str, Any]], show_detail: bool = False) -> None:
        """Print the rendered output."""
        raise NotImplementedError  # pragma: no cover

    def get_renderable(
        self, data: List[Dict[str, Any]], show_detail: bool = False
    ) -> RenderableType:
        """Get rendered visualizer."""
        raise NotImplementedError  # pragma: no cover

    def apply_styling(self, header: str, **kwargs) -> None:
        """Apply the styling."""
        self._styling_map[header] = kwargs

    def add_substitution_rule(self, before: str, after: Any) -> None:
        """Add a substitution rule."""
        self._substitution_rule[before] = after

    def _substitute(self, val: str) -> str:
        if self.substitute_exact_match_only:
            # Substitute only when `val` exactly matches to a rule.
            if val in self._substitution_rule:
                return self._substitution_rule[val]
        else:
            # Apply substitution for all matched substrings.
            for before, after in self._substitution_rule.items():
                if before in val:
                    return val.replace(before, after)
        return val


@dataclass
class TableFormatter(ListFormatter):
    """Table formatter for visualizing tabulated data."""

    caption: Optional[str] = None
    table: Optional[Table] = None

    def _init(self, show_detail: bool) -> None:
        self.table = Table(title=self.name, caption=self.caption, box=box.SIMPLE)
        self._make_header(show_detail)

    def render(self, data: List[Dict[str, Any]], show_detail: bool = False) -> None:
        """Print the rendered output."""
        self._build_table(data, show_detail)
        self._console.print(self.table)

    def get_renderable(
        self, data: List[Dict[str, Any]], show_detail: bool = False
    ) -> Table:
        """Get rendered visualizer."""
        self._build_table(data, show_detail)
        return cast(Table, self.table)

    def _build_table(self, data: List[Dict[str, Any]], show_detail: bool) -> None:
        self._init(show_detail)

        for d in data:
            info = [self._substitute(get_value(d, f)) for f in self.fields]
            if show_detail:
                info.extend(
                    [self._substitute(get_value(d, f)) for f in self.extra_fields]
                )
            cast(Table, self.table).add_row(*info)

    def _make_header(self, show_detail: bool) -> None:
        for header in self.headers:
            cast(Table, self.table).add_column(
                header, **self._styling_map.get(header, {})
            )

        if show_detail:
            for extra_header in self.extra_headers:
                cast(Table, self.table).add_column(extra_header)


@dataclass
class PanelFormatter(ListFormatter):
    """Panel formatter for visualizing information detail in a panel."""

    subtitle: Optional[str] = None
    panel: Optional[Panel] = None

    def render(
        self,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        show_detail: bool = False,
    ) -> None:
        """Print the rendered output."""
        if not isinstance(data, list):
            data = [data]
        self._build_panel(data, show_detail)
        self._console.print(self.panel)

    def get_renderable(
        self,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        show_detail: bool = False,
    ) -> Panel:
        """Get rendered visualizer."""
        if not isinstance(data, list):
            data = [data]
        self._build_panel(data, show_detail)
        return cast(Panel, self.panel)

    def _build_panel(self, data: List[Dict[str, Any]], show_detail: bool):
        headers = self.headers + self.extra_headers if show_detail else self.headers
        table = Table(box=None, show_header=False)
        table.add_column("k", style="dim bold")
        table.add_column("v")

        for d in data:
            info = [self._substitute(get_value(d, f)) for f in self.fields]
            if show_detail:
                info.extend(
                    [self._substitute(get_value(d, f)) for f in self.extra_fields]
                )
            for k, v in zip(headers, info):
                table.add_row(k, v)
        self.panel = Panel(table, title=self.name, subtitle=self.subtitle)


@dataclass
class Edge:
    """Edge of file traversal tree."""

    name: str
    size: int | None


def find_and_insert(parent: Tree, edges: List[Edge]) -> None:
    """Recursively construct a file tree."""
    if not edges:
        return

    is_dir = len(edges) > 1
    folder_style_prefix = "[bold magenta]:open_file_folder: "

    match = [
        tree
        for tree in parent.children
        if edges[0].name
        == str(tree.label).replace(folder_style_prefix, "").split(" ", maxsplit=1)[0]
    ]

    if match:
        tree = match[0]
    else:
        if is_dir:
            tree = parent.add(f"{folder_style_prefix}{edges[0].name}")
        else:
            text_filename = Text(edges[0].name, "green")
            file_size = edges[0].size or 0
            text_filename.append(f" ({decimal(file_size)})", "blue")
            tree = parent.add(text_filename)

    find_and_insert(tree, edges[1:])


class TreeFormatter(Formatter):
    """Tree formatter for visualizing file tree."""

    def __init__(self, name: str, root: str = ""):
        """Initialize tree formatter."""
        super().__init__(name=name)
        self._root = "/" + root.lstrip("/")
        self._panel: Optional[Panel] = None

    def render(self, data: List[Dict[str, Any]]) -> None:
        """Print the rendered output."""
        self._build_renderable(data)
        self._console.print(self._panel)

    def get_renderable(self, data: List[Dict[str, Any]]) -> Panel:
        """Get rendered visualizer."""
        self._build_renderable(data)
        return cast(Panel, self._panel)

    def _build_tree(self, data: List[Dict[str, Any]]) -> Tree:
        root = Tree("/")
        paths = [
            f"{d['path']}"
            if os.path.isabs(d["path"])
            else f"{os.path.join(self._root, d['path'])}"
            for d in data
        ]
        sizes = [d["size"] for d in data]
        for path, size in zip(paths, sizes):
            edges = []
            parts = path.split("/")[1:]
            for i, part in enumerate(parts):
                file_size = size if i == len(parts) - 1 else None
                edges.append(Edge(part, file_size))
            find_and_insert(root, edges)
        return root

    def _build_renderable(self, data: List[Dict[str, Any]]) -> None:
        root = self._build_tree(data)
        self._panel = Panel(root, title=self.name)


@dataclass
class JSONFormatter(Formatter):
    """JSON formatter for visualizing JSON data."""

    panel: Optional[Panel] = None

    def render(self, data: Dict[str, Any]) -> None:
        """Print the rendered output."""
        self._build_json(data)
        self._console.print(self.panel)

    def get_renderable(self, data: Dict[str, Any]) -> Panel:
        """Get rendered visualizer."""
        self._build_json(data)
        return cast(Panel, self.panel)

    def _build_json(self, data: Dict[str, Any]) -> None:
        self.panel = Panel(JSON(json.dumps(data)), title=self.name)
