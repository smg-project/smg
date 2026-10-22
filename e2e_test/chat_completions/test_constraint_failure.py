"""An engine that cannot continue a constrained request is one explicit error.

When the engine admits a request that carries a constraint (a ``response_format``
JSON schema here) and then cannot continue it under that constraint, the
worker ends the stream with one status every wire shares, and the gateway
answers it the same way whatever the engine: a 422 under the code
``constraint_failed`` (an ``invalid_request_error``), never a 200 with an empty
or truncated answer, and never a retry (the same tokens fail the same way on
every replay, each replay a prefill). A streamed response that the failure
cuts after its first tokens ends with the error event under the same code,
then ``[DONE]``.

The mock engine (realistic mode) reports the failure through its fault hook:
``fail?status=422&after_tokens=K`` cuts the next request's stream after K
tokens with the worker's constraint-failure status, as the wires produce it
when the engine attributes the failure to the grammar. Its other shape,
``fail?finish=error&after_tokens=K``, is the engine failing the request without
naming a cause: the engine's own report, which the gateway retries as it always
did, constraint or not.

Usage:
    E2E_RUNTIME=mock E2E_ENGINE=mock pytest e2e_test/chat_completions/test_constraint_failure.py -v
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "city",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def _arm_constraint_failure(mock_answers, after_tokens: int, count: int) -> None:
    """The mock worker's next ``count`` requests end with the constraint failure
    after ``after_tokens`` tokens."""
    response = httpx.post(
        f"{mock_answers.worker.admin_url}/admin/fault/all/fail",
        params={"status": 422, "after_tokens": after_tokens, "count": count},
        timeout=10.0,
    )
    response.raise_for_status()


def _arm_engine_error(mock_answers, after_tokens: int, count: int) -> None:
    """The mock worker's next ``count`` requests end as an engine that failed
    the request without naming a cause does: a ``Complete`` with finish reason
    ``error`` after ``after_tokens`` tokens (the engine's own report)."""
    response = httpx.post(
        f"{mock_answers.worker.admin_url}/admin/fault/all/fail",
        params={"finish": "error", "after_tokens": after_tokens, "count": count},
        timeout=10.0,
    )
    response.raise_for_status()


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
@pytest.mark.workers(extra_engine_args=["--engine", "realistic"])
@pytest.mark.gateway(extra_args=["--retry-max-retries", "5"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestConstraintFailure:
    """The gateway's answer for a constraint the engine could not continue the request under."""

    def test_the_answer_is_one_explicit_error_and_one_engine_call(
        self, setup_backend, mock_answers
    ):
        _, model, _, gateway = setup_backend
        before = _requests_served(mock_answers)
        _arm_constraint_failure(mock_answers, after_tokens=2, count=3)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Name a city."}],
            "max_tokens": 16,
            "response_format": _SCHEMA,
        }
        for _ in range(3):
            resp = httpx.post(f"{gateway.base_url}/v1/chat/completions", json=body, timeout=60.0)
            logger.info("status=%s body=%s", resp.status_code, resp.text[:300])
            assert resp.status_code == 422, resp.text[:300]
            error = resp.json()["error"]
            assert error["code"] == "constraint_failed", error
            assert error["type"] == "invalid_request_error", error
            assert error["message"].startswith(
                "the engine could not continue the request under its constraint"
            ), error
            assert resp.headers.get("x-smg-error-code") == "constraint_failed"
        # One engine request per client request: the retry budget of five
        # bought no replay, and every stream was the one the fault cut.
        assert _requests_served(mock_answers) - before == 3
        state = httpx.get(f"{mock_answers.worker.admin_url}/admin/requests", timeout=10.0).json()
        cut = sum(int(w.get("cut") or 0) for w in state.get("injected_failures", {}).values())
        assert cut >= 3, state.get("injected_failures")

    def test_a_cut_stream_ends_with_the_error_event_under_its_code(
        self, setup_backend, mock_answers
    ):
        _, model, _, gateway = setup_backend
        _arm_constraint_failure(mock_answers, after_tokens=2, count=1)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Name a city."}],
            "max_tokens": 16,
            "stream": True,
            "response_format": _SCHEMA,
        }
        with httpx.stream(
            "POST", f"{gateway.base_url}/v1/chat/completions", json=body, timeout=60.0
        ) as resp:
            assert resp.status_code == 200, resp.read()[:300]
            assert resp.headers.get("content-type", "").startswith("text/event-stream")
            payloads = [
                line[len("data: ") :] for line in resp.iter_lines() if line.startswith("data: ")
            ]
        logger.info("payloads=%s", payloads)
        assert payloads[-1] == "[DONE]", payloads
        event = json.loads(payloads[-2])
        assert event["error"]["code"] == "constraint_failed", event
        assert event["error"]["type"] == "invalid_request_error", event
        assert event["error"]["message"].startswith(
            "Stream error: the engine could not continue the request under its constraint"
        ), event
        chunks = [json.loads(p) for p in payloads[:-2]]
        assert chunks and all("choices" in chunk for chunk in chunks), payloads

    def test_an_engine_error_the_engine_does_not_attribute_is_retried_as_before(
        self, setup_backend, mock_answers
    ):
        # The same constrained request, the engine failing it without naming a
        # cause (its `error` finish, as a failed KV handoff or a pause ends a
        # request): the engine's own report, which the gateway replays on the
        # retry budget as it always did, never the constraint's 422. The one
        # armed failure is spent on the first attempt; the replay serves.
        _, model, _, gateway = setup_backend
        before = _requests_served(mock_answers)
        _arm_engine_error(mock_answers, after_tokens=2, count=1)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Name a city."}],
            "max_tokens": 16,
            "response_format": _SCHEMA,
        }
        resp = httpx.post(f"{gateway.base_url}/v1/chat/completions", json=body, timeout=60.0)
        logger.info("status=%s body=%s", resp.status_code, resp.text[:300])
        assert resp.status_code == 200, resp.text[:300]
        assert resp.json()["choices"][0]["finish_reason"] in ("stop", "length")
        assert resp.headers.get("x-smg-error-code") is None
        # Two engine requests for one client request: the failed attempt and
        # the replay that answered.
        assert _requests_served(mock_answers) - before == 2


def _requests_served(mock_answers) -> int:
    """The mock worker's count of generate requests its engine admitted."""
    state = httpx.get(f"{mock_answers.worker.admin_url}/admin/requests", timeout=10.0).json()
    return len(state.get("records") or [])
