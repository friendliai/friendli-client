# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test Catalog Client."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from uuid import uuid4

import pytest
import requests_mock

from periflow.client.catalog import CatalogClient
from periflow.enums import CatalogStatus
from periflow.errors import APIError


@pytest.fixture
def catalog_client() -> CatalogClient:
    return CatalogClient()


@pytest.mark.usefixtures("patch_auto_token_refresh")
@pytest.mark.usefixtures("user_project_group_context")
def test_get_catalog(
    requests_mock: requests_mock.Mocker, catalog_client: CatalogClient
):
    catalog_id = uuid4()
    resp_body = dict(
        id=str(catalog_id),
        organization_id=str(uuid4()),
        name="fpt",
        attributes={"model_type": "gpt"},
        tags=["llm", "fai"],
        format="orca",
        summary="This model is awesome!",
        description="This model is super smart!",
        use_count=100,
        status=CatalogStatus.ACTIVE,
        status_reason=None,
        deleted=False,
        deleted_at=None,
        files=[
            dict(
                name="model.h5",
                path="fpt/model.h5",
                mtime=datetime.utcnow().isoformat(),
                size=9999,
                created_at=datetime.utcnow().isoformat(),
            )
        ],
        created_at=datetime.utcnow().isoformat(),
    )

    url_template = deepcopy(catalog_client.url_template)
    url_template.attach_pattern("$catalog_id/")

    requests_mock.get(
        url_template.render(catalog_id=catalog_id),
        status_code=200,
        json=resp_body,
    )
    assert catalog_client.get_catalog(catalog_id=catalog_id) == resp_body

    requests_mock.get(
        url_template.render(catalog_id=catalog_id),
        status_code=404,
    )
    with pytest.raises(APIError):
        catalog_client.get_catalog(catalog_id=catalog_id)
