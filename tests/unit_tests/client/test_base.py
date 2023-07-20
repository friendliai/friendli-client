# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test BaseClient Service"""


from __future__ import annotations

from string import Template

import pytest
import requests_mock

import periflow
from periflow.client.base import Client, URLTemplate


@pytest.fixture
def base_url() -> str:
    return "https://test.periflow.com/"


def test_url_template_render(base_url: str):
    url_pattern = f"{base_url}test/"
    template = URLTemplate(Template(url_pattern))
    assert template.render() == url_pattern
    assert template.render(pk=1) == f"{url_pattern}1/"
    assert template.render(pk="abcd") == f"{url_pattern}abcd/"


def test_url_template_render_complex_pattern(base_url: str):
    url_pattern = f"{base_url}test/$test_id/job/"
    template = URLTemplate(Template(url_pattern))

    # Missing an url param
    with pytest.raises(KeyError):
        template.render()

    assert template.render(test_id=1) == f"{base_url}test/1/job/"
    assert template.render("abcd", test_id=1) == f"{base_url}test/1/job/abcd/"


def test_url_template_attach_pattern(base_url: str):
    url_pattern = f"{base_url}test/$test_id/job/"
    template = URLTemplate(Template(url_pattern))

    template.attach_pattern("$job_id/export/")

    with pytest.raises(KeyError):
        template.render(test_id=1)

    assert (
        template.render(test_id=1, job_id="abcd")
        == f"{base_url}test/1/job/abcd/export/"
    )
    assert (
        template.render(0, test_id=1, job_id="abcd")
        == f"{base_url}test/1/job/abcd/export/0/"
    )


def test_client_service_base(requests_mock: requests_mock.Mocker, base_url: str):
    url_pattern = f"{base_url}test/$test_id/job/"

    # Mock CRUD requests
    template = URLTemplate(Template(url_pattern))
    requests_mock.get(template.render(test_id=1), json=[{"data": "value"}])
    requests_mock.get(template.render("abcd", test_id=1), json={"data": "value"})
    requests_mock.post(
        template.render(test_id=1), json={"data": "value"}, status_code=201
    )
    requests_mock.patch(template.render("abcd", test_id=1), json={"data": "value"})
    requests_mock.delete(template.render("abcd", test_id=1), status_code=204)

    class TestClient(Client[int]):
        @property
        def url_path(self) -> Template:
            return Template(url_pattern)

    client = TestClient(test_id=1)

    periflow.api_key = "test-api-key"
    resp = client.list()
    assert resp.json() == [{"data": "value"}]
    assert resp.status_code == 200

    resp = client.retrieve("abcd")
    assert resp.json() == {"data": "value"}
    assert resp.status_code == 200

    resp = client.post()
    assert resp.json() == {"data": "value"}
    assert resp.status_code == 201

    resp = client.partial_update("abcd")
    assert resp.json() == {"data": "value"}
    assert resp.status_code == 200

    resp = client.delete("abcd")
    assert resp.status_code == 204
