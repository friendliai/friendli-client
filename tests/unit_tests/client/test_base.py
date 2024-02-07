# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Test BaseClient Service"""


from __future__ import annotations

from string import Template

import pytest
import requests_mock

import friendli
from friendli.client.base import Client, URLTemplate


@pytest.fixture
def base_url() -> str:
    return "https://test.friendli.com/"


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


@pytest.mark.parametrize("pagination", [True, False])
def test_client_service_base(
    requests_mock: requests_mock.Mocker, base_url: str, pagination: bool
):
    url_pattern = f"{base_url}test/$test_id/job/"

    # Mock CRUD requests
    template = URLTemplate(Template(url_pattern))
    if pagination:
        mock_list_resp = {
            "results": [
                {"data": "value"},
            ],
            "next_cursor": None,
        }
    else:
        mock_list_resp = [{"data": "value"}]
    requests_mock.get(template.render(test_id=1), json=mock_list_resp)
    requests_mock.get(template.render("abcd", test_id=1), json={"data": "value"})
    requests_mock.post(
        template.render(test_id=1), json={"data": "value"}, status_code=201
    )
    requests_mock.patch(template.render("abcd", test_id=1), json={"data": "value"})
    requests_mock.delete(template.render("abcd", test_id=1), status_code=204)

    class TestClient(Client[int]):
        @property
        def url_path(self) -> str:
            return url_pattern

    client = TestClient(test_id=1)

    friendli.token = "test-api-key"
    data = client.list(pagination=pagination)
    assert data == [{"data": "value"}]

    data = client.retrieve("abcd")
    assert data == {"data": "value"}

    data = client.post()
    assert data == {"data": "value"}

    data = client.partial_update("abcd")
    assert data == {"data": "value"}

    data = client.delete("abcd")
