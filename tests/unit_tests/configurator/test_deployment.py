# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test Deployment Configurator."""

from __future__ import annotations

import json
from contextlib import nullcontext
from tempfile import TemporaryFile

import pytest
import typer
from _pytest.fixtures import SubRequest

from periflow.configurator.deployment import DRCConfigurator
from periflow.errors import InvalidConfigError
from periflow.utils.testing import merge_dicts


@pytest.fixture
def stop_config(request: SubRequest) -> dict:
    return {"stop": ["a", "about", "above"]} if request.param else {}


@pytest.fixture
def stop_tokens_config(request: SubRequest) -> dict:
    return {"stop_tokens": {"tokens": [1, 2, 3, 4]}} if request.param else {}


@pytest.fixture
def bad_words_config(request: SubRequest) -> dict:
    return {"bad_words": ["F*ck", "Shxt"]} if request.param else {}


@pytest.fixture
def bad_word_tokens_config(request: SubRequest) -> dict:
    return {"bad_word_tokens": {"tokens": [199, 200]}} if request.param else {}


@pytest.fixture
def invalid_config(request: SubRequest) -> dict:
    return {"wrong": "field"} if request.param else {}


class TestDRCConfigurator:
    """Test `DRCConfigurator`."""

    @pytest.mark.parametrize("stop_config", [True, False], indirect=True)
    @pytest.mark.parametrize("stop_tokens_config", [True, False], indirect=True)
    @pytest.mark.parametrize("bad_words_config", [True, False], indirect=True)
    @pytest.mark.parametrize("bad_word_tokens_config", [True, False], indirect=True)
    @pytest.mark.parametrize("invalid_config", [True, False], indirect=True)
    def test_validation(
        self,
        stop_config: dict,
        stop_tokens_config: dict,
        bad_words_config: dict,
        bad_word_tokens_config: dict,
        invalid_config: dict,
    ):
        config = merge_dicts(
            [
                stop_config,
                stop_tokens_config,
                bad_words_config,
                bad_word_tokens_config,
                invalid_config,
            ]
        )
        ctx = nullcontext()
        if (
            (stop_config and stop_tokens_config)
            or (bad_words_config and bad_word_tokens_config)
            or not (
                stop_config
                or stop_tokens_config
                or bad_words_config
                or bad_word_tokens_config
            )
            or invalid_config
        ):
            ctx = pytest.raises(InvalidConfigError)
        with TemporaryFile(prefix="periflow-cli-unittest", mode="r+") as f:
            json.dump(config, f)
            f.seek(0)
            configurator = DRCConfigurator.from_file(f)
            with ctx:
                configurator.validate()
