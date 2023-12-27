# Copyright (c) 2022-present, FriendliAI Inc. All rights reserved.

"""Friendli CLI API Request Utilities."""

from __future__ import annotations

from requests.exceptions import HTTPError

from friendli.utils.url import discuss_url

DEFAULT_REQ_TIMEOUT = 30
MAX_RETRIES = 3


def decode_http_err(exc: HTTPError) -> str:
    """Decode HTTP error."""
    try:
        if exc.response.status_code == 500:
            error_str = f"Internal Server Error: Please contact to system admin via {discuss_url}"
        elif exc.response.status_code == 404:
            error_str = (
                "Not Found: The requested resource is not found. Please check it again. "
                f"If you cannot find out why this error occurs, please visit {discuss_url}."
            )
        else:
            response = exc.response
            detail_json = response.json()
            if "detail" in detail_json:
                error_str = f"Error Code: {response.status_code}\nDetail: {detail_json['detail']}"
            elif "error_description" in detail_json:
                error_str = (
                    f"Error Code: {response.status_code}\n"
                    f"Detail: {detail_json['error_description']}"
                )
            else:
                error_str = f"Error Code: {response.status_code}"
    except ValueError:
        error_str = exc.response.content.decode()

    return error_str
