"""Background mode of the Responses API (``background: true``).

A background request is answered at once with a ``queued`` Response object;
the generation runs behind the response id, which is polled with
``GET /v1/responses/{id}`` (``queued`` -> ``in_progress`` -> a terminal
status) and stopped with ``POST /v1/responses/{id}/cancel``. With
``stream: true`` the stream carries ``response.queued`` right after
``response.created``. A background response the client asks not to store
(``store`` false) is queued the same way: its record is polled through the run
and holds the result for a polling window, after which it is deleted.
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


def _poll_raw_until_terminal(
    gateway, response_id: str, timeout: float = 120.0, interval: float = 0.5
) -> dict:
    """Poll ``GET /v1/responses/{id}`` on the raw wire until a terminal status."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = httpx.get(
            f"{gateway.base_url}/v1/responses/{response_id}",
            headers={"Authorization": "Bearer not-used"},
            timeout=30.0,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in TERMINAL_STATUSES:
            return body
        time.sleep(interval)
    raise TimeoutError(f"response {response_id} not terminal after {timeout}s")


@pytest.mark.engine("sglang", "mock")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestBackgroundResponsesLocal:
    """Background mode against a local gRPC worker.

    On the mock engine each test states the answer first; the one the test cancels is
    paced by a delay per token, so the cancel lands while it is running.
    """

    def test_background_response_is_queued_then_completes(self, model, api_client, mock_answers):
        """The request answers ``queued`` at once; polling reaches ``completed``
        with the generated output under the same id."""
        mock_answers.enqueue("Autumn turns the leaves to gold. The air cools as the days shorten.")
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

    def test_cancel_stops_a_running_background_response(self, model, api_client, mock_answers):
        """Cancel answers the ``cancelled`` object; the stored record agrees and a
        second cancel answers the same object."""
        mock_answers.enqueue(" ".join(["lighthouse"] * 400), delay_ms=20)
        create_resp = api_client.responses.create(
            model=model,
            input="Write a long story about a lighthouse keeper, at least twenty paragraphs.",
            background=True,
            max_output_tokens=1024,
        )
        assert create_resp.status in ("queued", "in_progress")

        # On the mock, wait until the worker took the paced answer, so the
        # cancel lands on a running generation (one that arrives before the
        # request reached the worker drops the work unsent and leaves the
        # answer queued); a no-op on a real engine.
        deadline = time.monotonic() + 10
        while mock_answers.active and mock_answers.leftovers() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not mock_answers.leftovers(), "the mock worker did not take the paced answer"

        cancelled = api_client.responses.cancel(create_resp.id)
        assert cancelled.id == create_resp.id
        assert cancelled.status == "cancelled"
        assert cancelled.background is True
        assert cancelled.completed_at is not None and cancelled.completed_at > 0

        stored = api_client.responses.retrieve(response_id=create_resp.id)
        assert stored.status == "cancelled"

        again = api_client.responses.cancel(create_resp.id)
        assert again.status == "cancelled"

    def test_background_stream_starts_with_queued(self, model, api_client, mock_answers):
        """A background stream opens with created, queued, in_progress and ends
        with the terminal event; the object is stored under the stream's id."""
        mock_answers.enqueue("1, 2, 3, 4, 5.")
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


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed", "mock")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b", mock="openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestBackgroundResponsesGptOss:
    """Background mode on the reasoning-model path, which serves the request
    through its own pipeline: the same lifecycle, under the same id."""

    def test_background_response_is_queued_then_completes(self, model, api_client, mock_answers):
        mock_answers.enqueue("Autumn turns the leaves. The evenings come early.")
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

    def test_cancel_stops_a_running_background_response(self, model, api_client, mock_answers):
        # On the mock: an answer that takes seconds to stream, so the cancel
        # finds the response running.
        mock_answers.enqueue(
            " ".join(["The keeper climbed the stairs at dusk."] * 60), chunk_tokens=4, delay_ms=100
        )
        create_resp = api_client.responses.create(
            model=model,
            input="Write a long story about a lighthouse keeper, at least twenty paragraphs.",
            background=True,
            max_output_tokens=1024,
        )
        assert create_resp.status in ("queued", "in_progress")

        # On the mock, wait until the worker took the paced answer, so the
        # cancel lands on a running generation (one that arrives first drops
        # the work unsent and leaves the answer queued); a no-op on a real
        # engine.
        deadline = time.monotonic() + 10
        while mock_answers.active and mock_answers.leftovers() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not mock_answers.leftovers(), "the mock worker did not take the paced answer"

        cancelled = api_client.responses.cancel(create_resp.id)
        assert cancelled.id == create_resp.id
        assert cancelled.status == "cancelled"
        assert api_client.responses.retrieve(response_id=create_resp.id).status == "cancelled"

    def test_background_stream_starts_with_queued(self, model, api_client, mock_answers):
        mock_answers.enqueue("1, 2, 3, 4, 5.")
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


@pytest.mark.engine("sglang", "mock")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestBackgroundResponsesRawLocal:
    """The unstored background response and the refusals, read on the raw
    wire shape."""

    def test_background_without_store_is_queued_and_polled(self, setup_backend, mock_answers):
        """``background: true`` with ``store: false`` is queued like a stored
        one: the record is polled through the run and holds the result (the
        record is deleted after the polling window, which the test does not
        wait for)."""
        mock_answers.enqueue("Hi there!")
        _, model_path, _, gw = setup_backend
        resp = _post_responses(
            gw,
            {
                "model": model_path,
                "input": "hi",
                "background": True,
                "store": False,
                "max_output_tokens": 64,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] in ("queued", "in_progress"), body
        assert body["background"] is True
        assert body["store"] is False
        assert body["output"] == []

        final = _poll_raw_until_terminal(gw, body["id"])
        assert final["status"] in ("completed", "incomplete"), final
        assert final["background"] is True
        assert final["store"] is False
        texts = [
            part["text"]
            for item in final["output"]
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        ]
        assert "".join(texts), "the result is polled from the record"

    def test_cancel_of_a_finished_response_is_refused(self, setup_backend, mock_answers):
        """A response that already ended cannot be cancelled."""
        mock_answers.enqueue("Hi!")
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
