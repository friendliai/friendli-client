# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Suite CLI commands."""

from __future__ import annotations

from typer import Typer

from ..const import Panel
from ..typer_util import merge_typer
from .common_auth import group as common_auth_group
from .common_meta import group as common_meta_group
from .common_team import app as common_team_app
from .dedicated_endpoint import app as dedicated_endpoint_app
from .dedicated_model import app as dedicated_model_app
from .dedicated_project import app as dedicated_project_app

app = Typer()

merge_typer(app, common_auth_group)
app.add_typer(common_team_app, rich_help_panel=Panel.COMMON)
merge_typer(app, common_meta_group)

app.add_typer(dedicated_model_app, rich_help_panel=Panel.DEDICATED)
app.add_typer(dedicated_project_app, rich_help_panel=Panel.DEDICATED)
app.add_typer(dedicated_endpoint_app, rich_help_panel=Panel.DEDICATED)
