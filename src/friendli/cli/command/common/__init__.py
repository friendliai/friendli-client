# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli Suite Common commands."""

from __future__ import annotations

from typer import Typer

from ..const import COMMON_COMMANDS
from ..util import merge_typer
from .common_auth import group as common_auth_group
from .common_meta import group as common_meta_group
from .common_team import app as common_team_app

common_app = Typer()
common_team_group = Typer()
common_team_group.add_typer(common_team_app, rich_help_panel=COMMON_COMMANDS)

merge_typer(common_app, common_auth_group)
merge_typer(common_app, common_team_group)
merge_typer(common_app, common_meta_group)
