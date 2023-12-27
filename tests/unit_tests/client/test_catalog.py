# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test Catalog Client."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
import requests_mock

from friendli.client.catalog import CatalogClient
from friendli.enums import CatalogImportMethod
from friendli.errors import APIError


@pytest.fixture
def catalog_client() -> CatalogClient:
    return CatalogClient()


@pytest.fixture
def catalog_id() -> uuid4():
    return uuid4()


@pytest.fixture
def catalog_resp_body(catalog_id: UUID) -> dict[str, Any]:
    return dict(
        id=str(catalog_id),
        organization_id=str(uuid4()),
        name="fpt",
        attributes={"model_type": "gpt"},
        tags=["llm", "fai"],
        format="orca",
        summary="This model is awesome!",
        description="This model is super smart!",
        use_count=100,
        status="Active",
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


@pytest.mark.usefixtures("patch_safe_request")
@pytest.mark.usefixtures("user_project_group_context")
def test_get_catalog(
    requests_mock: requests_mock.Mocker,
    catalog_client: CatalogClient,
    catalog_id: UUID,
    catalog_resp_body: dict[str, Any],
):
    url_template = deepcopy(catalog_client.url_template)
    url_template.attach_pattern("$catalog_id/")

    requests_mock.get(
        url_template.render(catalog_id=catalog_id),
        status_code=200,
        json=catalog_resp_body,
    )
    assert catalog_client.get_catalog(catalog_id=catalog_id) == catalog_resp_body

    requests_mock.get(
        url_template.render(catalog_id=catalog_id),
        status_code=404,
    )
    with pytest.raises(APIError):
        catalog_client.get_catalog(catalog_id=catalog_id)


@pytest.mark.usefixtures("patch_safe_request")
@pytest.mark.usefixtures("user_project_group_context")
def test_list_catalog(
    requests_mock: requests_mock.Mocker,
    catalog_client: CatalogClient,
    catalog_resp_body: dict[str, Any],
):
    resp_body = {
        "results": [catalog_resp_body],
        "next_cursor": None,
    }

    requests_mock.get(
        catalog_client.url_template.render(),
        status_code=200,
        json=resp_body,
    )
    actual = catalog_client.list_catalogs(name=None, limit=10)
    assert actual == resp_body["results"]

    requests_mock.get(
        catalog_client.url_template.render(),
        status_code=400,
    )

    with pytest.raises(APIError):
        catalog_client.list_catalogs(name=None, limit=10)


@pytest.mark.usefixtures("patch_safe_request")
@pytest.mark.usefixtures("user_project_group_context")
def test_try_out_catalog(
    requests_mock: requests_mock.Mocker,
    catalog_client: CatalogClient,
    catalog_id: UUID,
    catalog_resp_body: dict[str, Any],
):
    catalog_id = uuid4()
    resp_body = dict(
        id=str(uuid4()),
        user_id=str(uuid4()),
        model_category="REF",
        job_id=None,
        name="sample-model",
        attributes={"model_type": "gpt"},
        iteration=None,
        tags=["llm"],
        catalog=catalog_resp_body,
        status="Active",
        status_reason=None,
        validation_status=None,
        validation_status_reason=None,
        created_at=datetime.utcnow().isoformat(),
        deleted=False,
        deleted_at=None,
        hard_deleted=False,
        hard_deleted_at=None,
    )

    url_template = deepcopy(catalog_client.url_template)
    url_template.attach_pattern("$catalog_id/try_out/")

    requests_mock.post(
        url_template.render(catalog_id=catalog_id),
        status_code=201,
        json=resp_body,
    )
    actual = catalog_client.try_out(
        catalog_id=catalog_id, name="model-ref", method=CatalogImportMethod.REF
    )
    assert actual == resp_body

    requests_mock.post(
        url_template.render(catalog_id=catalog_id),
        status_code=400,
    )
    with pytest.raises(APIError):
        catalog_client.try_out(
            catalog_id=catalog_id, name="model-ref", method=CatalogImportMethod.REF
        )
