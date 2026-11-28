"""The input-items list: the paging parameters a caller sets reach the route's query."""

from __future__ import annotations

from typing import Any

import pytest

from smg_client import AsyncSmgClient, SmgClient

BASE = "http://gateway.test"
ROUTE = f"{BASE}/v1/responses/resp_1/input_items"


def page(ids: list[str], has_more: bool) -> dict[str, Any]:
    """A list envelope as the route answers it."""
    return {
        "object": "list",
        "data": [{"id": item_id, "type": "message"} for item_id in ids],
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
        "has_more": has_more,
    }


def test_the_list_without_parameters_sends_no_query(httpx_mock):
    httpx_mock.add_response(url=ROUTE, json=page(["msg_1"], False))
    with SmgClient(base_url=BASE) as client:
        items = client.responses.input_items.list(response_id="resp_1")
    assert [item.id for item in items.data] == ["msg_1"]
    assert items.has_more is False
    assert str(httpx_mock.get_request().url) == ROUTE


@pytest.mark.parametrize(
    ("kwargs", "query"),
    [
        ({"limit": 5}, "limit=5"),
        ({"order": "asc"}, "order=asc"),
        ({"after": "msg_9"}, "after=msg_9"),
        (
            {"include": ["message.input_image.image_url"]},
            "include=message.input_image.image_url",
        ),
        (
            {"limit": 2, "order": "desc", "after": "msg_3", "include": ["a", "b"]},
            "limit=2&order=desc&after=msg_3&include=a&include=b",
        ),
    ],
)
def test_each_parameter_reaches_the_query_as_the_public_api_names_it(httpx_mock, kwargs, query):
    httpx_mock.add_response(url=f"{ROUTE}?{query}", json=page([], False))
    with SmgClient(base_url=BASE) as client:
        client.responses.input_items.list(response_id="resp_1", **kwargs)
    assert str(httpx_mock.get_request().url) == f"{ROUTE}?{query}"


def test_a_two_page_walk_follows_the_previous_page_last_id(httpx_mock):
    httpx_mock.add_response(url=f"{ROUTE}?limit=2", json=page(["msg_3", "msg_2"], True))
    httpx_mock.add_response(url=f"{ROUTE}?limit=2&after=msg_2", json=page(["msg_1"], False))
    with SmgClient(base_url=BASE) as client:
        first = client.responses.input_items.list(response_id="resp_1", limit=2)
        assert first.has_more is True
        second = client.responses.input_items.list(
            response_id="resp_1", limit=2, after=first.last_id
        )
    assert [item.id for item in first.data + second.data] == ["msg_3", "msg_2", "msg_1"]
    assert second.has_more is False


def test_the_compatibility_alias_keeps_the_route_defaults(httpx_mock):
    httpx_mock.add_response(url=ROUTE, json=page(["msg_1"], False))
    with SmgClient(base_url=BASE) as client:
        items = client.responses.list_input_items("resp_1")
    assert items.last_id == "msg_1"
    assert str(httpx_mock.get_request().url) == ROUTE


async def test_the_async_list_sends_the_same_query(httpx_mock):
    httpx_mock.add_response(url=f"{ROUTE}?limit=1&order=asc", json=page(["msg_1"], True))
    async with AsyncSmgClient(base_url=BASE) as client:
        items = await client.responses.input_items.list(response_id="resp_1", limit=1, order="asc")
    assert items.last_id == "msg_1"
    assert items.has_more is True
    assert str(httpx_mock.get_request().url) == f"{ROUTE}?limit=1&order=asc"
