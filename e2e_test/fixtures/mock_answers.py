"""The answers a test states for the mock engine before it sends a request.

Under ``E2E_RUNTIME=mock`` the workers are scripted mock workers
(``crates/mock_worker``): they answer with what was enqueued on their admin
API, in order, and fall back to a canned answer (counted as a miss) when
nothing was. A test that runs on the mock states the model's answer right
before the request, in the model's own syntax (the tool-call JSON the lane's
parser reads, a reasoning block, the text that holds the stop sequence), and
the gateway does the rest: tokenization, the wire, decoding, parsing.

On a real engine the fixture does nothing, so one test body serves both.

Each mock worker is its own process with its own queue, and a request's
worker is the gateway's choice. A class that runs one mock worker states its
answers in the order of its requests; a class that runs several (the decode
pool of a disaggregated class, where the prefill side answers the prompt's
completion alone) states each answer with the prompt it is for
(``prompt_contains``): the answer is filed on every worker and the one the
gateway picks serves it, whatever order the requests arrive in. The backend
setup registers the class's workers here (every wire, cached or not).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, overload

import httpx
import pytest
from infra import get_runtime
from infra.constants import WorkerType
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


def class_workers() -> list[Worker]:
    """The class's workers, every role: a test that reads one worker's own
    admin API (an encode worker's count of the completions it served, which
    the gateway does not count) finds it here."""
    return list(_CLASS_WORKERS)


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


HARMONY_FINAL_OPEN = "<|channel|>final<|message|>"
HARMONY_RETURN = "<|return|>"


@overload
def on_final_channel(output_text: str, *, closed: bool = True) -> str: ...


@overload
def on_final_channel(output_text: list[str], *, closed: bool = True) -> list[str]: ...


def on_final_channel(output_text: str | list[str], *, closed: bool = True) -> str | list[str]:
    """A plain text on a Harmony model's final channel, as the model writes
    an answer; a text that opens with a channel marker as it is (it states its
    channels itself). The channel is closed with the model's own end unless
    the answer was cut short (``closed=False``: a matched stop sequence, the
    token limit), when an engine leaves it open. Pieces keep their shape: the
    markers are pieces of their own."""
    pieces = [output_text] if isinstance(output_text, str) else list(output_text)
    if pieces and pieces[0].startswith("<|"):
        return output_text
    wrapped = [HARMONY_FINAL_OPEN, *pieces, *([HARMONY_RETURN] if closed else [])]
    return "".join(wrapped) if isinstance(output_text, str) else wrapped


def _ends_on_its_own(answer: dict[str, Any]) -> bool:
    """Whether a stated answer ends with the model's own end: not on a matched
    stop sequence, not on the token limit."""
    return answer.get("matched_stop") is None and answer.get("finish_reason") != "length"


@dataclass
class MockAnswers:
    """Enqueue answers on the class's mock workers."""

    # The workers that take the answers: the class's one worker, or the
    # decode workers of a disaggregated class. Empty on a real engine.
    workers: list[Worker] = field(default_factory=list)
    enqueued: int = 0
    # What the workers had served when the test began (``note_served``): the
    # hits since are the test's own.
    _served_before: int = 0

    @property
    def active(self) -> bool:
        return bool(self.workers)

    @property
    def worker(self) -> Worker | None:
        """The class's mock worker (the first of several: its model, its admin API)."""
        return self.workers[0] if self.workers else None

    @property
    def harmony(self) -> bool:
        """The class's model answers on the Harmony channels (a gpt-oss model:
        the gateway picks that pipeline by the model's name), so a stated text
        rides on the final channel unless the answer states its channels."""
        return self.worker is not None and "gpt-oss" in self.worker.model_id.lower()

    def enqueue(self, output_text: str | list[str] | None = None, **answer: Any) -> None:
        """State the mock worker's next answer.

        ``output_text`` is the model's output (one string, or the pieces it
        streams); the other keys are the mock's (``output_ids``,
        ``finish_reason``, ``matched_stop``, ``cached_tokens``,
        ``prompt_tokens``, ``logprobs``, ``delay_ms``, ``chunk_tokens``).
        ``prompt_contains`` files the answer for the first request whose
        prompt holds the text instead of the next request: what a test
        states when its requests run concurrently, or when the class runs
        several workers and the gateway picks one (the answer is filed on
        every worker; the one that serves it counts the hit). A class of
        several workers takes no answer without it. Once every stated answer
        has been served, the copies left on the workers that did not serve
        one are dropped before the next is filed, so a later prompt that
        carries an earlier one's text (a conversation's second turn) meets no
        sibling's leftover. On a Harmony model the text rides on the final
        channel (the mock's tokenizer turns the channel markers into the
        special tokens the gateway's Harmony pipeline reads), closed with the
        model's own end unless the answer ends on a ``matched_stop`` or on
        ``length``, when the channel stays open as an engine leaves it; a text
        that opens with a channel marker states its channels itself
        (``enqueue_harmony``). No-op on a real engine.
        """
        if not self.workers:
            return
        if output_text is not None:
            answer["output_text"] = (
                on_final_channel(output_text, closed=_ends_on_its_own(answer))
                if self.harmony
                else output_text
            )
        self._file(answer)

    def _file(self, answer: dict[str, Any]) -> None:
        """File a built answer on the class's workers (the checks and the count
        ``enqueue`` documents)."""
        if len(self.workers) > 1 and "prompt_contains" not in answer:
            raise ValueError(
                f"this class runs {len(self.workers)} mock workers and the gateway picks one: "
                "state the prompt the answer is for (enqueue(..., prompt_contains=...))"
            )
        if self.enqueued and self.served() == self.enqueued:
            self.clear()
        for worker in self.workers:
            response = httpx.post(f"{worker.admin_url}/admin/script/all", json=answer, timeout=10.0)
            response.raise_for_status()
        self.enqueued += 1

    def enqueue_harmony(
        self,
        *,
        final: str | None = None,
        analysis: str | None = None,
        tool_call: tuple[str, dict[str, Any] | str] | None = None,
        **answer: Any,
    ) -> None:
        """State the mock worker's next answer on the Harmony channels.

        ``analysis`` is the reasoning (the analysis channel), then either
        ``tool_call`` (a function's name and its JSON arguments on the
        commentary channel; the answer ends with the call) or ``final`` (the
        final channel). Each channel after the first opens a new assistant
        message, as the model writes them. No-op on a real engine.
        """
        if not self.workers:
            return
        if tool_call is not None and final is not None:
            raise ValueError(
                "state a tool call or a final text, not both (the answer ends with the call)"
            )
        segments: list[str] = []
        if analysis is not None:
            segments.append(f"<|channel|>analysis<|message|>{analysis}<|end|>")
        if tool_call is not None:
            name, arguments = tool_call
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            segments.append(
                f"<|channel|>commentary to=functions.{name} <|constrain|>json"
                f"<|message|>{arguments}<|call|>"
            )
        elif final is not None:
            segments.append(f"{HARMONY_FINAL_OPEN}{final}{HARMONY_RETURN}")
        if not segments:
            raise ValueError("state an analysis, a final text or a tool call")
        self.enqueue("<|start|>assistant".join(segments), **answer)

    def enqueue_scores(self, logprobs: list[float]) -> None:
        """State the mock worker's next scoring answer: the logprob of each token
        id the request asks a score for, in the request's order, on a completion
        of no output tokens (what a prefill-only scoring request reads: the
        gateway's Decisions stage sends one per question on the SGLang wire, its
        candidate labels' ids in the question's order). An answer of no tokens
        has no channel, so a Harmony model's scoring answer is filed as it is,
        with an empty ``output_text``. No-op on a real engine.
        """
        if not self.workers:
            return
        self._file({"output_text": "", "token_id_logprobs": list(logprobs)})

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
        if self.harmony:
            output_text = on_final_channel(output_text, closed=_ends_on_its_own(answer))
        ids, pieces = _pieces(get_model_spec(self.worker.model_id)["model"], output_text)
        self.enqueue(pieces, output_ids=ids, logprobs=[logprob] * len(ids), **answer)

    def served(self) -> int:
        """The answers the workers served since ``note_served``: one hit per
        choice a request took (``n`` for an ``n>1`` request), on whichever
        worker served it."""
        return self._hits() - self._served_before

    def note_served(self) -> None:
        """Count the answers served from here on (the test's own)."""
        self._served_before = self._hits()

    def _hits(self) -> int:
        return sum(
            int(httpx.get(f"{w.admin_url}/admin/script/all", timeout=10.0).json().get("hits") or 0)
            for w in self.workers
        )

    def leftovers(self) -> int:
        """Stated answers no request took (negative: more served than stated,
        a prompt text that matched more than one request or a request that
        asked for several choices)."""
        if not self.workers:
            return 0
        return self.enqueued - self.served()

    def clear(self) -> None:
        """Drop what is still queued or filed on every worker (the hit counts stay)."""
        for worker in self.workers:
            httpx.delete(f"{worker.admin_url}/admin/script/all", timeout=10.0)


@pytest.fixture
def mock_answers(setup_backend):
    """The class's mock workers' answer queues, or an inactive no-op elsewhere.

    After the test, an answer the test enqueued and never consumed fails it,
    as does an answer served more often than it was stated: the script and
    the requests no longer match.
    """
    if get_runtime() != "mock":
        yield MockAnswers()
        return
    workers = [w for w in _CLASS_WORKERS if w.admin_url is not None]
    # A disaggregated class: the answers are the decode side's; its prefill
    # side answers the prompt's completion alone.
    decodes = [w for w in workers if w.worker_type == WorkerType.DECODE]
    if decodes:
        workers = decodes
    if not workers:
        pytest.fail(
            "mock engine: this backend has no mock worker with an admin API "
            "(see the module doc of fixtures/mock_answers.py)"
        )
    answers = MockAnswers(workers=workers)
    answers.note_served()
    try:
        yield answers
    finally:
        leftovers = answers.leftovers()
        answers.clear()
        if leftovers > 0:
            pytest.fail(f"mock engine: {leftovers} scripted answer(s) no request took")
        if leftovers < 0:
            pytest.fail(
                f"mock engine: {-leftovers} more answer(s) served than stated: a prompt text "
                "matched more than one request, or a request asked for several choices"
            )
