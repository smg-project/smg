"""What the vLLM servicer can tell about its engine's load from the requests it
forwards: queued token-work, generation throughput and the prefix-cache hit
rate, per data-parallel rank: the ``GetLoads`` fields the gateway's
expected-wait score reads and vLLM's own ``SchedulerStats`` do not carry
(they report running, waiting and KV usage). Without them the gateway scores
a vLLM worker on defaults and routes it unlike its peers.

The servicer sees every request it submits and the rank serving it, the
engine's first output for it (with the prompt and cached token counts) and
every token streamed, so, for each rank:

- queued token-work is the uncached prompt tokens of the requests the rank
  has not started: of the submitted requests without a first output, the
  youngest ``num_waiting_reqs`` (the engine admits FCFS, so the older ones are
  in prefill), each prompt discounted by the rank's recent hit rate since its
  own cached count is unknown until it starts;
- generation throughput is the tokens the rank streamed over the last
  :data:`THROUGHPUT_WINDOW` seconds;
- the hit rate is cached over prompt tokens across the rank's last
  :data:`HIT_RATE_SAMPLES` first outputs.

A rank keeps its own figures: a report that put the whole servicer's on its
first rank read the other ranks as idle while they served. These are the
field semantics the SGLang servicer reports from its scheduler and the mock
engine from its queue; the Rust servicer keeps the same bookkeeping
(``crates/engine_servicer/src/load_tracker.rs``). Engine-free, so it is
unit-tested without vLLM installed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import NamedTuple

from smg_grpc_proto import vllm_engine_pb2
from smg_grpc_proto.generated import common_pb2

from smg_grpc_servicer.load_record import engine_load_from

THROUGHPUT_WINDOW = 2.0
HIT_RATE_SAMPLES = 64


class LoadEstimate(NamedTuple):
    """The three fields, as ``GetLoads`` reports them for one rank."""

    queued_token_work: int = 0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0


class _Rank:
    """One rank's recent first outputs and streamed tokens."""

    def __init__(self) -> None:
        # (prompt tokens, cached tokens) of recent first outputs.
        self.hits: deque[tuple[int, int]] = deque()
        self.prompt_sum = 0
        self.cached_sum = 0
        # (time, tokens) streamed within the window.
        self.generated: deque[tuple[float, int]] = deque()
        self.generated_sum = 0

    def trim(self, now: float) -> None:
        while self.generated and now - self.generated[0][0] > THROUGHPUT_WINDOW:
            _, tokens = self.generated.popleft()
            self.generated_sum -= tokens

    def hit_rate(self) -> float:
        if self.prompt_sum <= 0:
            return 0.0
        return min(1.0, max(0.0, self.cached_sum / self.prompt_sum))


class _Pending:
    """A submitted request without a first output yet."""

    __slots__ = ("prompt_tokens", "rank", "request_id")

    def __init__(self, request_id: str, prompt_tokens: int, rank: int | None) -> None:
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        # The rank it went to; ``None`` until the engine has routed it.
        self.rank = rank


class LoadTracker:
    """Per-servicer load bookkeeping, kept per rank; thread-safe, every call
    is cheap. ``rank_of`` names the rank a request submitted without one
    went to, once the engine has routed it (``None`` until then); without
    it, such a request's queued prompt counts toward no rank."""

    def __init__(
        self, clock=time.monotonic, rank_of: Callable[[str], int | None] | None = None
    ) -> None:
        self._clock = clock
        self._rank_of = rank_of
        self._lock = threading.Lock()
        # Submitted, no first output yet, oldest first.
        self._pending: deque[_Pending] = deque()
        # The ranks that have started a request or streamed a token.
        self._ranks: dict[int, _Rank] = {}

    def submitted(self, request_id: str, prompt_tokens: int, *, rank: int | None) -> None:
        """A request with ``prompt_tokens`` was handed to ``rank``, or to the
        engine's choice of rank when ``None`` (read through ``rank_of``)."""
        with self._lock:
            self._pending.append(
                _Pending(
                    request_id, max(0, int(prompt_tokens)), None if rank is None else int(rank)
                )
            )

    def first_output(
        self, request_id: str, prompt_tokens: int, cached_tokens: int, *, rank: int
    ) -> None:
        """``rank``'s first output for a request: it has started, and its
        prompt had ``cached_tokens`` of ``prompt_tokens`` in that rank's
        prefix cache."""
        with self._lock:
            self._drop_pending(request_id)
            prompt_tokens = max(0, int(prompt_tokens or 0))
            if prompt_tokens == 0:
                return
            cached_tokens = min(max(0, int(cached_tokens or 0)), prompt_tokens)
            figures = self._ranks.setdefault(int(rank), _Rank())
            figures.hits.append((prompt_tokens, cached_tokens))
            figures.prompt_sum += prompt_tokens
            figures.cached_sum += cached_tokens
            if len(figures.hits) > HIT_RATE_SAMPLES:
                prompt, cached = figures.hits.popleft()
                figures.prompt_sum -= prompt
                figures.cached_sum -= cached

    def generated(self, tokens: int, *, rank: int, at: float | None = None) -> None:
        """``rank`` streamed ``tokens`` to the client just now (or ``at``)."""
        tokens = int(tokens)
        if tokens <= 0:
            return
        now = self._clock() if at is None else at
        with self._lock:
            figures = self._ranks.setdefault(int(rank), _Rank())
            figures.generated.append((now, tokens))
            figures.generated_sum += tokens
            figures.trim(now)

    def pending_rank(self, request_id: str) -> int | None:
        """The rank a submitted request without a first output went to, as
        far as ``rank_of`` has told (read now if not yet): what the tracker
        learned while the request was queued, for a caller that finds the
        engine's own record of it gone by its first output."""
        with self._lock:
            for entry in self._pending:
                if entry.request_id == request_id:
                    return self._rank_of_pending(entry)
        return None

    def finished(self, request_id: str) -> None:
        """A request ended (or was aborted) without the engine starting it."""
        with self._lock:
            self._drop_pending(request_id)

    def pending(self) -> int:
        """Submitted requests the engine has not started."""
        with self._lock:
            return len(self._pending)

    def estimate(
        self, num_waiting_reqs: int, *, rank: int, now: float | None = None
    ) -> LoadEstimate:
        """``rank``'s estimate now, given the engine's own count of that
        rank's waiting requests."""
        now = self._clock() if now is None else now
        with self._lock:
            figures = self._ranks.get(int(rank))
            if figures is None:
                hit_rate, gen_throughput = 0.0, 0.0
            else:
                figures.trim(now)
                hit_rate = figures.hit_rate()
                gen_throughput = figures.generated_sum / THROUGHPUT_WINDOW
            pending = [
                entry.prompt_tokens
                for entry in self._pending
                if self._rank_of_pending(entry) == int(rank)
            ]
            waiting = min(max(0, int(num_waiting_reqs or 0)), len(pending))
            queued = 0.0
            for prompt in pending[len(pending) - waiting :]:
                queued += prompt * (1.0 - hit_rate)
            return LoadEstimate(
                queued_token_work=int(round(queued)),
                gen_throughput=gen_throughput,
                cache_hit_rate=hit_rate,
            )

    def _rank_of_pending(self, entry: _Pending) -> int | None:
        """The entry's rank, read through ``rank_of`` while the engine's
        choice is not known yet (kept once it is)."""
        if entry.rank is None and self._rank_of is not None:
            entry.rank = self._rank_of(entry.request_id)
        return entry.rank

    def _drop_pending(self, request_id: str) -> None:
        for index, entry in enumerate(self._pending):
            if entry.request_id == request_id:
                del self._pending[index]
                return


def scheduler_load_fields(
    num_running: int,
    num_waiting: int,
    kv_usage: float,
    estimate: LoadEstimate,
    *,
    dp_rank: int = 0,
    max_total_num_tokens: int = 0,
    max_running_requests: int = 0,
) -> dict:
    """The ``SchedulerLoad`` fields for one rank: vLLM's own counts plus the
    tracker's estimate for that rank, with ``num_used_tokens`` and
    ``utilization`` derived from the KV usage as the SGLang servicer derives
    them."""
    kv_usage = max(0.0, float(kv_usage or 0.0))
    fields = {
        "dp_rank": int(dp_rank),
        "num_running_reqs": int(num_running),
        "num_waiting_reqs": int(num_waiting),
        "num_total_reqs": int(num_running) + int(num_waiting),
        "token_usage": kv_usage,
        "utilization": kv_usage,
        "num_waiting_uncached_tokens": int(estimate.queued_token_work),
        "gen_throughput": float(estimate.gen_throughput),
        "cache_hit_rate": float(estimate.cache_hit_rate),
    }
    if max_total_num_tokens > 0:
        fields["max_total_num_tokens"] = int(max_total_num_tokens)
        fields["num_used_tokens"] = int(round(kv_usage * max_total_num_tokens))
    if max_running_requests > 0:
        fields["max_running_requests"] = int(max_running_requests)
    return fields


def load_record(
    num_running: int,
    num_waiting: int,
    kv_usage: float,
    estimate: LoadEstimate,
    *,
    dp_rank: int = 0,
    max_total_num_tokens: int = 0,
    max_running_requests: int = 0,
) -> common_pb2.EngineLoad:
    """The load record the KV-event stream carries for one rank, from the
    figures :func:`scheduler_load_fields` reports on ``GetLoads``: the two
    agree field for field. The queued token-work is this servicer's own
    estimate for the rank, so every rank's record carries it, as on the
    poll."""
    load = vllm_engine_pb2.SchedulerLoad(
        **scheduler_load_fields(
            num_running,
            num_waiting,
            kv_usage,
            estimate,
            dp_rank=dp_rank,
            max_total_num_tokens=max_total_num_tokens,
            max_running_requests=max_running_requests,
        )
    )
    return engine_load_from(load, queued_token_work=True)
