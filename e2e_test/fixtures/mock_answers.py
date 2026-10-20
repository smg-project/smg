"""The answers a test states for the mock engine before it sends a request.

Under ``E2E_RUNTIME=mock`` the workers are scripted mock workers
(``crates/mock_worker``): they answer with what was enqueued on their admin
API, in order, and fall back to a canned answer (counted as a miss) when
nothing was. A test that runs on the mock states the model's answer right
before the request, in the model's own syntax (the tool-call JSON the lane's
parser reads, a reasoning block, the text that holds the stop sequence), and
the gateway does the rest: tokenization, the wire, decoding, parsing.

On a real engine the fixture does nothing, so one test body serves both.

A class on the mock runs one mock worker: each worker is its own process
with its own queue, and a request's worker is the gateway's choice, so two
workers could not be given answers in the request's order. The backend setup
registers the class's workers here (every wire, cached or not).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from infra import get_runtime
from infra.worker import Worker

logger = logging.getLogger(__name__)

# The workers of the class being served, set by the backend setup fixture for
# the life of the class.
_CLASS_WORKERS: list[Worker] = []


def set_class_workers(workers: list[Worker]) -> None:
    """The backend setup names the class's workers (every wire)."""
    _CLASS_WORKERS[:] = list(workers)


def clear_class_workers() -> None:
    _CLASS_WORKERS.clear()


@dataclass
class MockAnswers:
    """Enqueue answers on the class's mock worker."""

    worker: Worker | None = None
    enqueued: int = 0

    @property
    def active(self) -> bool:
        return self.worker is not None

    def enqueue(self, output_text: str | list[str] | None = None, **answer: Any) -> None:
        """State the mock worker's next answer.

        ``output_text`` is the model's output (one string, or the pieces it
        streams); the other keys are the mock's (``output_ids``,
        ``finish_reason``, ``matched_stop``, ``cached_tokens``,
        ``prompt_tokens``, ``logprobs``, ``delay_ms``, ``chunk_tokens``).
        No-op on a real engine.
        """
        if self.worker is None:
            return
        if output_text is not None:
            answer["output_text"] = output_text
        response = httpx.post(
            f"{self.worker.admin_url}/admin/script/all", json=answer, timeout=10.0
        )
        response.raise_for_status()
        self.enqueued += 1

    def leftovers(self) -> int:
        """Queued answers no request took."""
        if self.worker is None:
            return 0
        state = httpx.get(f"{self.worker.admin_url}/admin/script/all", timeout=10.0).json()
        return int(state.get("queued") or 0)

    def clear(self) -> None:
        if self.worker is not None:
            httpx.delete(f"{self.worker.admin_url}/admin/script/all", timeout=10.0)


@pytest.fixture
def mock_answers(setup_backend):
    """The class's mock worker's answer queue, or an inactive no-op elsewhere.

    After the test, an answer the test enqueued and never consumed fails it:
    the script and the request no longer match.
    """
    if get_runtime() != "mock":
        yield MockAnswers()
        return
    workers = [w for w in _CLASS_WORKERS if w.admin_url is not None]
    if len(workers) != 1:
        pytest.fail(
            "mock engine: a class runs one mock worker with an admin API, "
            f"this backend has {len(workers)} (see the module doc of fixtures/mock_answers.py)"
        )
    answers = MockAnswers(worker=workers[0])
    try:
        yield answers
    finally:
        leftovers = answers.leftovers()
        answers.clear()
        if leftovers:
            pytest.fail(f"mock engine: {leftovers} scripted answer(s) no request took")
