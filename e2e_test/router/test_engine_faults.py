"""The shapes a client sees when an engine refuses or stalls, through the edge
of a running gateway, stated on the mock engine's fault hooks.

A refusal before admission (the mock's ``fail?status=S``) reaches the gateway
as the gRPC code it maps back: 503 ``unavailable``, 429 ``resource_exhausted``,
500 ``internal``, all retryable by default. The retry layer replays such a
refusal on another worker while its budget lasts; past the budget the worker's
status is the client's answer under ``start_generation_failed``. A stall
longer than the opt-in call bound (``--grpc-request-timeout-secs``) is the 504
``call_upstream_timeout``, terminal for the retry layer. A paused engine holds
the requests it was given, so behind a concurrency cap of one the admission
queue refuses the next request with its 503 (``admission_queue_timeout``) and
serves again once the engine resumes. A worker that fails past the breaker's
threshold is excluded while the other serves.

Usage:
    E2E_RUNTIME=mock E2E_ENGINE=mock pytest e2e_test/router/test_engine_faults.py -v
"""

from __future__ import annotations

import logging
import threading
import time

import httpx
import pytest

logger = logging.getLogger(__name__)

_MODEL = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
_RETRY_BUDGET = 3
# The class's call bound and the stall that outlasts it (seconds).
_CALL_BOUND_SECS = 3
_STALL_SECS = 8


def _arm(worker, **params) -> None:
    """The worker's fault hook: ``fail`` with the given query."""
    response = httpx.post(f"{worker.admin_url}/admin/fault/all/fail", params=params, timeout=10.0)
    response.raise_for_status()


def _clear(worker) -> None:
    _arm(worker, status=0)


def _hook(worker, name: str) -> None:
    response = httpx.post(f"{worker.admin_url}/admin/fault/all/{name}", timeout=10.0)
    response.raise_for_status()


def _refusals(worker) -> int:
    """The requests the worker's hook refused before admission."""
    state = httpx.get(f"{worker.admin_url}/admin/requests", timeout=10.0).json()
    return sum(int(w.get("failed") or 0) for w in state.get("injected_failures", {}).values())


def _served(worker) -> int:
    """The requests the worker's engine admitted."""
    state = httpx.get(f"{worker.admin_url}/admin/requests", timeout=10.0).json()
    return len(state.get("records") or [])


def _stalls(worker) -> int:
    """The requests the worker's hook held before admission."""
    state = httpx.get(f"{worker.admin_url}/admin/fault/all", timeout=10.0).json()
    return sum(int(w.get("stalled_total") or 0) for w in state.get("workers", []))


def _metric(gateway, name: str, needle: str = "") -> float | None:
    """The first sample of the family ``name`` whose labels hold ``needle``;
    ``None`` when no sample matches, so an absent series never reads as 0."""
    text = httpx.get(f"{gateway.metrics_url}/metrics", timeout=10.0).text
    for line in text.splitlines():
        if (line.startswith(name + "{") or line.startswith(name + " ")) and needle in line:
            return float(line.rsplit(" ", 1)[1])
    return None


def _chat(gateway, model: str, timeout: float = 60.0) -> httpx.Response:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 8,
    }
    return httpx.post(f"{gateway.base_url}/v1/chat/completions", json=body, timeout=timeout)


def _refused(resp: httpx.Response, status: int, code: str) -> None:
    """The answer is the status under the code, in the body and the header."""
    assert resp.status_code == status, resp.text[:300]
    error = resp.json()["error"]
    assert error["code"] == code, error
    assert resp.headers.get("x-smg-error-code") == code, dict(resp.headers)


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model(_MODEL)
@pytest.mark.e2e
@pytest.mark.workers(count=2, extra_engine_args=["--engine", "realistic"])
@pytest.mark.gateway(
    extra_args=[
        "--retry-max-retries",
        str(_RETRY_BUDGET),
        "--grpc-request-timeout-secs",
        str(_CALL_BOUND_SECS),
        # Out of reach here: the retry budget, not the breaker, is what this
        # class reads (the breaker has the class below).
        "--cb-failure-threshold",
        "1000",
    ]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestEngineRefusals:
    """A refusal or a stall of one engine, two workers behind the gateway."""

    def test_a_refusal_under_the_retry_budget_is_replayed_on_the_other_worker(
        self, setup_backend, mock_answers
    ):
        _, model, _, gateway = setup_backend
        first, second = mock_answers.workers
        refusals_before = _refusals(first)
        first_served_before = _served(first)
        served_before = first_served_before + _served(second)
        # One refusal on one worker: whichever request the round robin sends
        # it meets the refusal and is replayed on the other worker, so every
        # request is served once and the refused attempt left no record.
        _arm(first, status=503, count=1)
        try:
            statuses = []
            while _refusals(first) == refusals_before and len(statuses) < 4:
                statuses.append(_chat(gateway, model).status_code)
        finally:
            _clear(first)
        assert statuses and all(status == 200 for status in statuses), statuses
        assert _refusals(first) - refusals_before == 1
        assert _served(first) + _served(second) - served_before == len(statuses)
        # The worker that refused admitted nothing: the replay went to the other.
        assert _served(first) - first_served_before == 0, (
            "the refused request was replayed on the worker that refused it"
        )

    def test_a_refusal_past_the_retry_budget_is_the_worker_s_status(
        self, setup_backend, mock_answers
    ):
        _, model, _, gateway = setup_backend
        workers = mock_answers.workers
        # Every worker refuses with the status for the whole test and the
        # status reaches the client as the gateway maps it, under the code of
        # a request the worker never started. A 503 or a 429 is replayed
        # until the budget is spent; a 500 is the worker's own answer for the
        # request, replayed once on the other worker and then final.
        for status, attempts in ((503, _RETRY_BUDGET), (429, _RETRY_BUDGET), (500, 2)):
            refusals_before = sum(_refusals(w) for w in workers)
            for worker in workers:
                _arm(worker, status=status, secs=60)
            try:
                resp = _chat(gateway, model)
            finally:
                for worker in workers:
                    _clear(worker)
            logger.info("status=%s -> %s %s", status, resp.status_code, resp.text[:200])
            _refused(resp, status, "start_generation_failed")
            assert sum(_refusals(w) for w in workers) - refusals_before == attempts, (
                f"one refusal per attempt on a {status}"
            )

    def test_a_stall_past_the_call_bound_is_a_504_and_no_replay(self, setup_backend, mock_answers):
        _, model, _, gateway = setup_backend
        workers = mock_answers.workers
        served_before = sum(_served(w) for w in workers)
        stalls_before = sum(_stalls(w) for w in workers)
        # The next request of every worker is held before admission for
        # longer than the call bound: the gateway drops the call at the bound.
        for worker in workers:
            _arm(worker, status=0, stall_ms=_STALL_SECS * 1000, count=1)
        started = time.perf_counter()
        try:
            resp = _chat(gateway, model)
        finally:
            for worker in workers:
                _clear(worker)
        elapsed = time.perf_counter() - started
        logger.info("stall -> %s %s after %.2fs", resp.status_code, resp.text[:200], elapsed)
        _refused(resp, 504, "call_upstream_timeout")
        assert _CALL_BOUND_SECS - 0.5 <= elapsed < _STALL_SECS, elapsed
        # Terminal: one attempt stalled, so the bound's answer was not replayed
        # on the other worker (a replay would have stalled there too).
        assert sum(_stalls(w) for w in workers) - stalls_before == 1, (
            "the timed-out call was replayed"
        )
        # The stalled request was never admitted, read once its stall is over:
        # the dropped call must not reach the engine when the hold ends.
        time.sleep(max(0.0, started + _STALL_SECS + 1 - time.perf_counter()))
        assert sum(_served(w) for w in workers) == served_before


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model(_MODEL)
@pytest.mark.e2e
@pytest.mark.workers(count=2, extra_engine_args=["--engine", "realistic"])
@pytest.mark.gateway(
    extra_args=[
        "--retry-max-retries",
        str(_RETRY_BUDGET),
        # One request at a time past admission; the next waits one second.
        "--max-concurrent-requests",
        "1",
        "--queue-size",
        "4",
        "--queue-timeout-secs",
        "1",
        # Two refusals open a worker's breaker, and it stays open for the
        # rest of the class.
        "--cb-failure-threshold",
        "2",
        "--cb-timeout-duration-secs",
        "600",
    ]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestPausedEngineAndBreaker:
    """A paused engine behind the admission queue, then a worker past the breaker."""

    def test_the_admission_queue_refuses_behind_a_paused_engine_and_serves_after_the_resume(
        self, setup_backend, mock_answers
    ):
        _, model, _, gateway = setup_backend
        workers = mock_answers.workers
        held: dict[str, httpx.Response | httpx.HTTPError] = {}

        def hold() -> None:
            try:
                held["response"] = _chat(gateway, model, timeout=120.0)
            except httpx.HTTPError as error:
                held["error"] = error

        for worker in workers:
            _hook(worker, "pause")
        first = threading.Thread(target=hold)
        try:
            # The first request takes the one admission permit and waits on
            # the paused engine; the second waits in the admission queue and
            # is refused at its timeout. The permit is read from the gateway's
            # in-flight gauge, not guessed from a sleep: sent before the first
            # request holds it, the second would take it and wait on the
            # paused engine until the client's timeout.
            first.start()
            deadline = time.monotonic() + 10
            while (_metric(gateway, "smg_admission_inflight") or 0.0) < 1:
                assert time.monotonic() < deadline, (
                    "the held request never took the admission permit"
                )
                time.sleep(0.05)
            started = time.perf_counter()
            resp = _chat(gateway, model)
            waited = time.perf_counter() - started
            logger.info("queued -> %s %s after %.2fs", resp.status_code, resp.text[:200], waited)
            _refused(resp, 503, "admission_queue_timeout")
            assert resp.headers.get("retry-after"), dict(resp.headers)
            assert 0.8 <= waited < 5, waited
        finally:
            for worker in workers:
                _hook(worker, "resume")
        first.join(timeout=120.0)
        assert "error" not in held, held["error"]
        assert held["response"].status_code == 200, held["response"].text[:300]
        assert _chat(gateway, model).status_code == 200

    def test_a_worker_past_the_breaker_threshold_is_excluded_while_the_other_serves(
        self, setup_backend, mock_answers
    ):
        _, model, _, gateway = setup_backend
        failing, serving = mock_answers.workers
        refusals_before = _refusals(failing)
        served_before = _served(serving)
        _arm(failing, status=503, secs=120)
        try:
            statuses = [_chat(gateway, model).status_code for _ in range(6)]
        finally:
            _clear(failing)
        assert statuses == [200] * 6, statuses
        # Two refusals opened the breaker; the requests after them never
        # reached the failing worker.
        assert _refusals(failing) - refusals_before == 2
        assert _served(serving) - served_before == 6
        # The series of each worker must be there: an absent one reads None.
        assert _metric(gateway, "smg_worker_cb_state", f':{failing.port}"') == 1.0
        assert _metric(gateway, "smg_worker_cb_state", f':{serving.port}"') == 0.0
