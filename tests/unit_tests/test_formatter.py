# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

# Copyright (C) 2021 FriendliAI

"""Test Client Service"""

from __future__ import annotations

import pytest
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from friendli.formatter import (
    JSONFormatter,
    PanelFormatter,
    TableFormatter,
    TreeFormatter,
    get_value,
)


@pytest.fixture
def table_formatter() -> TableFormatter:
    return TableFormatter(
        name="Personal Info",
        fields=["required.name", "email", "age"],
        headers=["Name", "Email", "Age"],
        extra_fields=["job", "active"],
        extra_headers=["Occupation", "Active"],
        caption="This table shows user's personal info",
    )


@pytest.fixture
def panel_formatter() -> PanelFormatter:
    return PanelFormatter(
        name="Personal Info",
        fields=["required.name", "email", "age"],
        headers=["Name", "Email", "Age"],
        extra_fields=["job", "active"],
        extra_headers=["Occupation", "Active"],
        subtitle="This table shows user's personal info",
    )


@pytest.fixture
def tree_formatter() -> TreeFormatter:
    return TreeFormatter(name="Files")


@pytest.fixture
def json_formatter() -> JSONFormatter:
    return JSONFormatter(name="Metadata")


def test_get_value():
    data = {
        "k1": {"k2": {"k3": "v1", "k4": "v2"}, "k5": "v3"},
        "k6": "v4",
        "k7": [{"k8": "v5"}],
    }

    assert get_value(data, "k1.k2.k3") == "v1"
    assert get_value(data, "k1.k2.k4") == "v2"
    assert get_value(data, "k1.k5") == "v3"
    assert get_value(data, "k6") == "v4"
    assert get_value(data, "k7[0].k8") == "v5"
    assert get_value(data, "k7[-1].k8") == "v5"


def test_table_formatter(
    table_formatter: TableFormatter, capsys: pytest.CaptureFixture
):
    data = [
        {
            "required": {"name": "koo"},
            "email": "koo@friendli.ai",
            "age": 26,
            "job": "historian",
            "active": True,
        },
        {
            "required": {"name": "kim"},
            "email": "kim@friendli.ai",
            "age": 28,
            "job": "scientist",
            "active": False,
        },
    ]
    table = table_formatter.get_renderable(data)
    assert isinstance(table, Table)
    table_formatter.render(data)
    out = capsys.readouterr().out
    assert "Active" not in out
    assert "Occupation" not in out

    table_formatter.add_substitution_rule("True", "Yes")
    table_formatter.add_substitution_rule("False", "No")
    table_formatter.apply_styling("Active", style="blue")
    table = table_formatter.get_renderable(data, show_detail=True)
    assert isinstance(table, Table)
    table_formatter.render(data, show_detail=True)
    out = capsys.readouterr().out
    assert "Active" in out
    assert "Occupation" in out
    assert "Yes" in out
    assert "No" in out


def test_panel_formatter(
    panel_formatter: PanelFormatter, capsys: pytest.CaptureFixture
):
    data = [
        {
            "required": {"name": "koo"},
            "email": "koo@friendli.ai",
            "age": 26,
            "job": "historian",
            "active": True,
        },
        {
            "required": {"name": "kim"},
            "email": "kim@friendli.ai",
            "age": 28,
            "job": "scientist",
            "active": False,
        },
    ]
    panel = panel_formatter.get_renderable(data)
    assert isinstance(panel, Panel)
    panel_formatter.render(data)
    out = capsys.readouterr().out
    assert "Active" not in out
    assert "Occupation" not in out

    panel_formatter.add_substitution_rule("True", "Yes")
    panel_formatter.add_substitution_rule("False", "No")
    panel_formatter.apply_styling("Active", style="blue")
    panel_formatter.get_renderable(data, show_detail=True)
    assert isinstance(panel, Panel)
    panel_formatter.render(data, show_detail=True)
    out = capsys.readouterr().out
    assert "Active" in out
    assert "Occupation" in out
    assert "Yes" in out
    assert "No" in out


def test_tree_formatter(tree_formatter: TreeFormatter, capsys: pytest.CaptureFixture):
    data = [
        {"path": "a", "size": 4},
        {"path": "dir1/b", "size": 1024},
        {"path": "dir1/dir2/c", "size": 1024 * 1024},
    ]
    tree = tree_formatter._build_tree(data)
    assert isinstance(tree, Tree)
    assert "/" in tree.label
    assert len(tree.children) == 2
    assert "a" in tree.children[1].label
    assert "dir1" in tree.children[1].label
    assert len(tree.children[1].children)
    assert "b" in tree.children[1].children[0].label
    assert "dir2" in tree.children[1].children[1].label
    assert len(tree.children[1].children[1].children) == 1
    assert "c" in tree.children[1].children[1].children[0].label

    panel = tree_formatter.get_renderable(data)
    assert isinstance(panel, Panel)
    tree_formatter.render(data)
    out = capsys.readouterr().out
    assert "a" in out
    assert "dir1" in out
    assert "b" in out
    assert "dir2" in out
    assert "c" in out


def test_json_formatter(json_formatter: JSONFormatter, capsys: pytest.CaptureFixture):
    data = {"k1": "v1", "k2": "v2"}
    panel = json_formatter.get_renderable(data)
    assert isinstance(panel, Panel)
    json_formatter.render(data)
    out = capsys.readouterr().out
    assert "k1" in out
    assert "v1" in out
    assert "k2" in out
    assert "v2" in out
