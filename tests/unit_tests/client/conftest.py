# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Common fixtures for client testing."""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_s3_presigned_url() -> str:
    return "https://my-example-bucket.s3.us-east-1.amazonaws.com/my-example-object?AWSAccessKeyId=AKIAEFGHIJKLMNOPQ123"
