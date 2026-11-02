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
import os
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from infra import get_runtime
from infra.model_specs import get_model_spec
from infra.worker import Worker
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# The workers of the class being served, set by the backend setup fixture for
# the life of the class.
_CLASS_WORKERS: list[Worker] = []


def set_class_workers(workers: list[Worker]) -> None:
    """The backend setup names the class's workers (every wire)."""
    _CLASS_WORKERS[:] = list(workers)


def clear_class_workers() -> None:
    _CLASS_WORKERS.clear()


# The tokenizers that encoded an answer, by model directory.
_TOKENIZERS: dict[str, Tokenizer] = {}


def _tokenizer(model_dir: str) -> Tokenizer:
    """The model's ``tokenizer.json``, loaded once per directory."""
    tokenizer = _TOKENIZERS.get(model_dir)
    if tokenizer is None:
        path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path}: no tokenizer file; the mock engine's classes read the model's "
                "files from ROUTER_LOCAL_MODEL_PATH/<model id>"
            )
        tokenizer = Tokenizer.from_file(path)
        _TOKENIZERS[model_dir] = tokenizer
    return tokenizer


def _pieces(model_dir: str, text: str) -> tuple[list[int], list[str]]:
    """``text`` as the ids the model's ``tokenizer.json`` gives it, no special
    tokens (the encode the mock worker does for an answer stated as text), and
    one piece of the text per id, so a text wire carries one logprob per piece.

    A piece runs from where its token starts in ``text`` to where the next one
    starts (decoding an id on its own loses a sentencepiece token's leading
    space), so the pieces concatenate back to ``text`` for every tokenizer whose
    offsets run forward; a character split over two ids leaves the second
    piece empty. Refused when the text has no tokens or the offsets do not
    partition it.
    """
    encoding = _tokenizer(model_dir).encode(text, add_special_tokens=False)
    if not encoding.ids:
        raise ValueError(f"{text!r}: no tokens to state a logprob for")
    bounds = [0] + [start for start, _ in encoding.offsets[1:]] + [len(text)]
    pieces = [text[start:end] for start, end in zip(bounds, bounds[1:])]
    if "".join(pieces) != text:
        raise ValueError(
            f"{text!r}: the tokenizer's offsets {encoding.offsets} do not partition the text; "
            "state this answer as pieces (enqueue([...], output_ids=[...], logprobs=[...]))"
        )
    return encoding.ids, pieces


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

    def enqueue_with_logprobs(self, output_text: str, logprob: float = -0.1, **answer: Any) -> None:
        """State the mock worker's next answer with a logprob per token.

        The text is stated as the ids the model's ``tokenizer.json`` gives it
        (the files the worker serves, the encode the mock does itself for a
        text answer), the piece of the text each id covers, and ``logprob`` on
        each: a token wire carries the ids, a text wire the pieces, one logprob
        per unit on either. A request that asks for logprobs then reads one per
        token; for ``top_logprobs`` the mock lists the answer's own distinct
        tokens as the alternatives, so an answer states at least as many as
        the request asks for. No-op on a real engine.
        """
        if self.worker is None:
            return
        ids, pieces = _pieces(get_model_spec(self.worker.model_id)["model"], output_text)
        self.enqueue(pieces, output_ids=ids, logprobs=[logprob] * len(ids), **answer)

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
