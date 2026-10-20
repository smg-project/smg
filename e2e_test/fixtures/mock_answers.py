"""The answers a test states for the mock engine before it sends a request.

Under ``E2E_RUNTIME=mock`` the workers are scripted mock workers
(``crates/mock_worker``): they answer with what was enqueued on their admin
API, in order, and fall back to a canned answer (counted as a miss) when
nothing was. A test that runs on the mock states the model's answer right
before the request, in the model's own syntax (the tool-call JSON the lane's
parser reads, a reasoning block, the text that holds the stop sequence), and
the gateway does the rest: tokenization, the wire, decoding, parsing.

On a real engine the fixture does nothing, so one test body serves both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from infra import get_runtime
from infra.worker import Worker
from infra.worker_pool import get_pool

logger = logging.getLogger(__name__)


@dataclass
class MockAnswers:
    """Enqueue answers on the mock workers of the current class."""

    workers: list[Worker] = field(default_factory=list)
    enqueued: int = 0

    @property
    def active(self) -> bool:
        return bool(self.workers)

    def enqueue(self, output_text: str | list[str] | None = None, **answer: Any) -> None:
        """State the next answer of every mock worker of the class.

        ``output_text`` is the model's output (one string, or the pieces it
        streams); the other keys are the mock's (``output_ids``,
        ``finish_reason``, ``matched_stop``, ``cached_tokens``,
        ``prompt_tokens``, ``logprobs``, ``delay_ms``, ``chunk_tokens``). The
        answer goes to every worker's shared queue, so whichever worker the
        gateway routes the request to answers with it. No-op on a real engine.
        """
        if not self.active:
            return
        if output_text is not None:
            answer["output_text"] = output_text
        for worker in self.workers:
            response = httpx.post(f"{worker.admin_url}/admin/script/all", json=answer, timeout=10.0)
            response.raise_for_status()
        self.enqueued += 1

    def leftovers(self) -> dict[str, int]:
        """Queued answers no request took, per worker admin URL."""
        left = {}
        for worker in self.workers:
            state = httpx.get(f"{worker.admin_url}/admin/script/all", timeout=10.0).json()
            if state.get("queued"):
                left[worker.admin_url or "?"] = int(state["queued"])
        return left

    def clear(self) -> None:
        for worker in self.workers:
            httpx.delete(f"{worker.admin_url}/admin/script/all", timeout=10.0)


@pytest.fixture
def mock_answers(setup_backend):
    """The class's mock workers' answer queue, or an inactive no-op elsewhere.

    After the test, an answer the test enqueued and never consumed fails it:
    the script and the request no longer match.
    """
    if get_runtime() != "mock":
        yield MockAnswers()
        return
    workers = [w for w in get_pool().current_workers() if w.admin_url is not None]
    answers = MockAnswers(workers=workers)
    if not workers:
        pytest.fail("mock engine: the backend has no mock worker with an admin API to script")
    try:
        yield answers
    finally:
        leftovers = answers.leftovers()
        answers.clear()
        if leftovers:
            pytest.fail(f"mock engine: scripted answers no request took: {leftovers}")
