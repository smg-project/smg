"""What the vLLM servicer can tell about its engine's load from the requests it
forwards: queued token-work, generation throughput and the prefix-cache hit
rate, the ``GetLoads`` fields the gateway's expected-wait score reads and
vLLM's own ``SchedulerStats`` do not carry (they report running, waiting and
KV usage). Without them the gateway scores a vLLM worker on defaults and
routes it unlike its peers.

The servicer sees every request it submits, the engine's first output for it
(with the prompt and cached token counts) and every token streamed, so:

- queued token-work is the uncached prompt tokens of the requests the engine
  has not started: of the submitted requests without a first output, the
  youngest ``num_waiting_reqs`` (the engine admits FCFS, so the older ones are
  in prefill), each prompt discounted by the recent hit rate since its own
  cached count is unknown until it starts;
- generation throughput is the tokens streamed over the last
  :data:`THROUGHPUT_WINDOW` seconds;
- the hit rate is cached over prompt tokens across the last
  :data:`HIT_RATE_SAMPLES` first outputs.

These are the field semantics the SGLang servicer reports from its scheduler
and the mock engine from its queue; the Rust servicer keeps the same
bookkeeping (``crates/engine_servicer/src/load_tracker.rs``). Engine-free, so
it is unit-tested without vLLM installed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import NamedTuple

THROUGHPUT_WINDOW = 2.0
HIT_RATE_SAMPLES = 64


class LoadEstimate(NamedTuple):
    """The three fields, as ``GetLoads`` reports them."""

    queued_token_work: int = 0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0


class LoadTracker:
    """Per-servicer load bookkeeping; thread-safe, every call is cheap."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        # (request_id, prompt_tokens), oldest first: submitted, no first output yet.
        self._pending: deque[tuple[str, int]] = deque()
        # (prompt tokens, cached tokens) of recent first outputs.
        self._hits: deque[tuple[int, int]] = deque()
        self._prompt_sum = 0
        self._cached_sum = 0
        # (time, tokens) streamed within the window.
        self._generated: deque[tuple[float, int]] = deque()
        self._generated_sum = 0

    def submitted(self, request_id: str, prompt_tokens: int) -> None:
        """A request with ``prompt_tokens`` was handed to the engine."""
        with self._lock:
            self._pending.append((request_id, max(0, int(prompt_tokens))))

    def first_output(self, request_id: str, prompt_tokens: int, cached_tokens: int) -> None:
        """The engine's first output for a request: it has started, and its
        prompt had ``cached_tokens`` of ``prompt_tokens`` in the prefix cache."""
        with self._lock:
            self._drop_pending(request_id)
            prompt_tokens = max(0, int(prompt_tokens or 0))
            if prompt_tokens == 0:
                return
            cached_tokens = min(max(0, int(cached_tokens or 0)), prompt_tokens)
            self._hits.append((prompt_tokens, cached_tokens))
            self._prompt_sum += prompt_tokens
            self._cached_sum += cached_tokens
            if len(self._hits) > HIT_RATE_SAMPLES:
                prompt, cached = self._hits.popleft()
                self._prompt_sum -= prompt
                self._cached_sum -= cached

    def generated(self, tokens: int, at: float | None = None) -> None:
        """``tokens`` were streamed to the client just now (or ``at``)."""
        tokens = int(tokens)
        if tokens <= 0:
            return
        now = self._clock() if at is None else at
        with self._lock:
            self._generated.append((now, tokens))
            self._generated_sum += tokens
            self._trim(now)

    def finished(self, request_id: str) -> None:
        """A request ended (or was aborted) without the engine starting it."""
        with self._lock:
            self._drop_pending(request_id)

    def pending(self) -> int:
        """Submitted requests the engine has not started."""
        with self._lock:
            return len(self._pending)

    def estimate(self, num_waiting_reqs: int, now: float | None = None) -> LoadEstimate:
        """The estimate now, given the engine's own count of waiting requests."""
        now = self._clock() if now is None else now
        with self._lock:
            self._trim(now)
            hit_rate = self._hit_rate()
            waiting = min(max(0, int(num_waiting_reqs or 0)), len(self._pending))
            queued = 0.0
            for _, prompt in list(self._pending)[len(self._pending) - waiting :]:
                queued += prompt * (1.0 - hit_rate)
            return LoadEstimate(
                queued_token_work=int(round(queued)),
                gen_throughput=self._generated_sum / THROUGHPUT_WINDOW,
                cache_hit_rate=hit_rate,
            )

    def _drop_pending(self, request_id: str) -> None:
        for index, (pending_id, _) in enumerate(self._pending):
            if pending_id == request_id:
                del self._pending[index]
                return

    def _trim(self, now: float) -> None:
        while self._generated and now - self._generated[0][0] > THROUGHPUT_WINDOW:
            _, tokens = self._generated.popleft()
            self._generated_sum -= tokens

    def _hit_rate(self) -> float:
        if self._prompt_sum <= 0:
            return 0.0
        return min(1.0, max(0.0, self._cached_sum / self._prompt_sum))


def scheduler_load_fields(
    num_running: int,
    num_waiting: int,
    kv_usage: float,
    estimate: LoadEstimate,
    *,
    max_total_num_tokens: int = 0,
    max_running_requests: int = 0,
) -> dict:
    """The ``SchedulerLoad`` fields for one rank: vLLM's own counts plus the
    tracker's estimate, with ``num_used_tokens`` and ``utilization`` derived
    from the KV usage as the SGLang servicer derives them."""
    kv_usage = max(0.0, float(kv_usage or 0.0))
    fields = {
        "dp_rank": 0,
        "num_running_reqs": int(num_running),
        "num_waiting_reqs": int(num_waiting),
        "num_waiting_uncached_tokens": int(estimate.queued_token_work),
        "num_total_reqs": int(num_running) + int(num_waiting),
        "token_usage": kv_usage,
        "utilization": kv_usage,
        "gen_throughput": float(estimate.gen_throughput),
        "cache_hit_rate": float(estimate.cache_hit_rate),
    }
    if max_total_num_tokens > 0:
        fields["max_total_num_tokens"] = int(max_total_num_tokens)
        fields["num_used_tokens"] = int(round(kv_usage * max_total_num_tokens))
    if max_running_requests > 0:
        fields["max_running_requests"] = int(max_running_requests)
    return fields


class RankLoadTrackers:
    """Keep request estimates in the same global rank space as scheduler stats.

    Unassigned requests on a DP frontend cannot be attributed until the engine
    exposes their chosen rank. They must not contribute to any rank's estimate.
    """

    def __init__(self, ranks, *, clock=time.monotonic):
        self._trackers = {rank: LoadTracker(clock=clock) for rank in ranks}

    def for_rank(self, rank: int | None) -> LoadTracker | None:
        if rank is None and len(self._trackers) == 1:
            return next(iter(self._trackers.values()))
        return self._trackers.get(rank)
