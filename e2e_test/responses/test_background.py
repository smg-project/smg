"""Background mode of the Responses API (``background: true``).

A background request is answered at once with a ``queued`` Response object;
the generation runs behind the response id, which is polled with
``GET /v1/responses/{id}`` (``queued`` -> ``in_progress`` -> a terminal
status) and stopped with ``POST /v1/responses/{id}/cancel``. With
``stream: true`` the stream carries ``response.queued`` right after
``response.created``. A background response has to be stored (``store`` true
or omitted): an unstored one could never be polled, so the request is refused.
"""

from __future__ import annotations

import logging
import time

import httpx
import pytest

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = ("completed", "incomplete", "failed", "cancelled")


def _poll_until_terminal(client, response_id: str, timeout: float = 120.0, interval: float = 0.5):
    """Poll ``GET /v1/responses/{id}`` until the object reaches a terminal status.

    Returns the final object and the statuses seen along the way, so a test
    can assert on the transitions it observed.
    """
    seen: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.responses.retrieve(response_id=response_id)
        assert resp.id == response_id
        if not seen or seen[-1] != resp.status:
            seen.append(resp.status)
        if resp.status in TERMINAL_STATUSES:
            return resp, seen
        time.sleep(interval)
    raise TimeoutError(f"response {response_id} still {seen[-1:]} after {timeout}s")


def _post_responses(gateway, body: dict, timeout: float = 120.0) -> httpx.Response:
    return httpx.post(
        f"{gateway.base_url}/v1/responses",
        json=body,
        headers={"Authorization": "Bearer not-used", "Content-Type": "application/json"},
        timeout=timeout,
    )


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestBackgroundResponsesLocal:
    """Background mode against a local gRPC worker."""

    def test_background_response_is_queued_then_completes(self, model, api_client):
        """The request answers ``queued`` at once; polling reaches ``completed``
        with the generated output under the same id."""
        create_resp = api_client.responses.create(
            model=model,
            input="Write two sentences about autumn.",
            background=True,
            max_output_tokens=64,
        )
        assert create_resp.id is not None
        assert create_resp.error is None
        assert create_resp.status in ("queued", "in_progress")
        assert create_resp.background is True
        assert create_resp.output == []
        assert create_resp.usage is None

        final, seen = _poll_until_terminal(api_client, create_resp.id)
        logger.info("background %s went through %s", create_resp.id, seen)
        assert final.status in ("completed", "incomplete"), f"{final.status}: {final.error}"
        assert final.background is True
        assert len(final.output_text) > 0
        assert final.usage is not None and final.usage.output_tokens > 0
        assert final.completed_at is not None and final.completed_at > 0

        input_items = api_client.responses.input_items.list(response_id=create_resp.id)
        assert input_items.data, "the input items are stored with the queued record"

    def test_cancel_stops_a_running_background_response(self, model, api_client):
        """Cancel answers the ``cancelled`` object; the stored record agrees and a
        second cancel answers the same object."""
        create_resp = api_client.responses.create(
            model=model,
            input="Write a long story about a lighthouse keeper, at least twenty paragraphs.",
            background=True,
            max_output_tokens=1024,
        )
        assert create_resp.status in ("queued", "in_progress")

        cancelled = api_client.responses.cancel(create_resp.id)
        assert cancelled.id == create_resp.id
        assert cancelled.status == "cancelled"
        assert cancelled.background is True
        assert cancelled.completed_at is not None and cancelled.completed_at > 0

        stored = api_client.responses.retrieve(response_id=create_resp.id)
        assert stored.status == "cancelled"

        again = api_client.responses.cancel(create_resp.id)
        assert again.status == "cancelled"

    def test_background_stream_starts_with_queued(self, model, api_client):
        """A background stream opens with created, queued, in_progress and ends
        with the terminal event; the object is stored under the stream's id."""
        stream = api_client.responses.create(
            model=model,
            input="Count from 1 to 5.",
            background=True,
            stream=True,
            max_output_tokens=32,
        )
        events = list(stream)
        types = [event.type for event in events]
        assert types[:3] == ["response.created", "response.queued", "response.in_progress"], types
        assert events[0].response.status == "queued"
        assert events[0].response.background is True
        assert types[-1] in ("response.completed", "response.incomplete"), types[-1]

        response_id = events[0].response.id
        assert events[-1].response.id == response_id
        stored = api_client.responses.retrieve(response_id=response_id)
        assert stored.status == events[-1].response.status
        assert stored.background is True


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestBackgroundResponsesGptOss:
    """Background mode on the reasoning-model path, which serves the request
    through its own pipeline: the same lifecycle, under the same id."""

    def test_background_response_is_queued_then_completes(self, model, api_client):
        create_resp = api_client.responses.create(
            model=model,
            input="Write two sentences about autumn.",
            background=True,
            max_output_tokens=256,
        )
        assert create_resp.error is None
        assert create_resp.status in ("queued", "in_progress")
        assert create_resp.background is True
        assert create_resp.output == []

        final, seen = _poll_until_terminal(api_client, create_resp.id)
        logger.info("background %s went through %s", create_resp.id, seen)
        assert final.status in ("completed", "incomplete"), f"{final.status}: {final.error}"
        assert final.background is True
        assert len(final.output) > 0, "the terminal object carries the generated items"
        assert final.completed_at is not None and final.completed_at > 0

    def test_cancel_stops_a_running_background_response(self, model, api_client):
        create_resp = api_client.responses.create(
            model=model,
            input="Write a long story about a lighthouse keeper, at least twenty paragraphs.",
            background=True,
            max_output_tokens=1024,
        )
        assert create_resp.status in ("queued", "in_progress")

        cancelled = api_client.responses.cancel(create_resp.id)
        assert cancelled.id == create_resp.id
        assert cancelled.status == "cancelled"
        assert api_client.responses.retrieve(response_id=create_resp.id).status == "cancelled"

    def test_background_stream_starts_with_queued(self, model, api_client):
        stream = api_client.responses.create(
            model=model,
            input="Count from 1 to 5.",
            background=True,
            stream=True,
            max_output_tokens=256,
        )
        events = list(stream)
        types = [event.type for event in events]
        assert types[:3] == ["response.created", "response.queued", "response.in_progress"], types
        assert events[0].response.status == "queued"
        assert types[-1] in ("response.completed", "response.incomplete"), types[-1]

        response_id = events[0].response.id
        assert events[-1].response.id == response_id
        stored = api_client.responses.retrieve(response_id=response_id)
        assert stored.status == events[-1].response.status


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestBackgroundResponsesRawLocal:
    """The refusals, read on the raw wire shape."""

    def test_background_requires_store(self, setup_backend):
        """``background: true`` with ``store: false`` is refused before anything
        runs: there would be nothing to poll."""
        _, model_path, _, gw = setup_backend
        resp = _post_responses(
            gw,
            {"model": model_path, "input": "hi", "background": True, "store": False},
        )
        assert resp.status_code == 400, resp.text
        error = resp.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "store" in error["message"]

    def test_cancel_of_a_finished_response_is_refused(self, setup_backend):
        """A response that already ended cannot be cancelled."""
        _, model_path, _, gw = setup_backend
        created = _post_responses(
            gw, {"model": model_path, "input": "Say hi", "max_output_tokens": 16}
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["status"] in ("completed", "incomplete"), body["status"]
        assert body["background"] is False

        cancel = httpx.post(
            f"{gw.base_url}/v1/responses/{body['id']}/cancel",
            json={},
            headers={"Authorization": "Bearer not-used"},
            timeout=30.0,
        )
        assert cancel.status_code == 400, cancel.text
        assert cancel.json()["error"]["code"] == "response_already_completed"
