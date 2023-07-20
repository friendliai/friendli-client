# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
import requests_mock

import periflow
from periflow.utils.url import get_training_uri


@pytest.fixture
def patch_auto_token_refresh(requests_mock: requests_mock.Mocker):
    periflow.api_key = "fake-api-key"
    requests_mock.post(get_training_uri("token/refresh"))


@pytest.fixture
def user_project_group_context():
    with patch(
        "periflow.client.base.UserRequestMixin.get_current_user_id",
        return_value=uuid.UUID("22222222-2222-2222-2222-222222222222"),
    ), patch(
        "periflow.client.base.get_current_group_id",
        return_value=uuid.UUID("00000000-0000-0000-0000-000000000000"),
    ), patch(
        "periflow.client.base.get_current_project_id",
        return_value=uuid.UUID("11111111-1111-1111-1111-111111111111"),
    ):
        yield
