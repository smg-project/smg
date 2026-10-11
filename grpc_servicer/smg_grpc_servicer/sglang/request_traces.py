"""The request traces the SGLang plugin keeps on the Rust servicer's wire.

Behind SGLang's own frontend the tokenizer manager starts each request's
trace from the HTTP request's headers (``init_trace_ctx`` on the request's
time stats, from the W3C ``traceparent``/``tracestate``), hands the scheduler
a pickled copy of those stats on the tokenized request (the scheduler's own
``Req`` takes them over, so its slices join the same trace) and finishes the
request's root span when the last output arrives. On the Rust servicer's
msgpack wire the frontend is Rust: it sends the Router's context in the
request's metadata frame (see ``decode_request_frames``), and this module
fills the tokenizer manager's role for the trace inside the scheduler
process: start from the context, hand the scheduler its copy, finish on the
terminal output.

Nothing here runs unless the engine was started with tracing (SGLang's
``--enable-trace``): without it the metadata frame is read and dropped.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

#: The metadata frame's key for the caller's W3C trace context.
TRACE_HEADERS_KEY = "trace_headers"


def tracing_enabled() -> bool:
    """Whether this process initialized SGLang's tracing (the scheduler does
    when the engine was started with ``--enable-trace``)."""
    try:
        from sglang.srt.observability.trace import get_global_tracing_enabled
    except ImportError:  # an SGLang without the switch: nothing to trace into
        return False
    return bool(get_global_tracing_enabled())


def _api_server_time_stats() -> Any:
    """The frontend-role time stats for one request, as the tokenizer manager
    creates them."""
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
    from sglang.srt.runtime_context import get_disagg

    return APIServerReqTimeStats(disagg_mode=DisaggregationMode(get_disagg().disaggregation_mode))


class RequestTraces:
    """The traces of the requests in flight on this wire, by request id: the
    frontend-role time stats whose root span the Router's context parents."""

    def __init__(self, enabled: bool | None = None) -> None:
        self._enabled = enabled
        self._stats: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = tracing_enabled()
        return self._enabled

    def __len__(self) -> int:
        return len(self._stats)

    def attach(self, request: Any, trace_headers: dict[str, str] | None) -> bool:
        """Start ``request``'s trace from the caller's context and give the
        scheduler its copy of the stats on the request (``time_stats``), as the
        tokenizer manager does; returns whether a trace was started. Nothing
        without a context, with tracing off, or for a request id already in
        flight (the scheduler refuses the duplicate)."""
        if not trace_headers or not self.enabled:
            return False
        rid = str(getattr(request, "rid", "") or "")
        if not rid or rid in self._stats:
            return False
        try:
            stats = _api_server_time_stats()
            stats.init_trace_ctx(rid, getattr(request, "bootstrap_room", None), dict(trace_headers))
            if not getattr(stats.trace_ctx, "tracing_enable", False):
                return False
            # The root span opens here; the Router tokenized upstream, so the
            # tokenize slice the stats open with ends at once.
            stats.set_created_time()
            stats.set_tokenize_finish_time()
            # The scheduler gets what the tokenizer manager's send gives it: a
            # pickled copy, which rebuilds its own thread context under the
            # root span; the root span stays here until the request ends.
            request.time_stats = pickle.loads(pickle.dumps(stats))
        except Exception:  # tracing never costs the request
            logger.exception("request trace: could not start the trace of %s", rid)
            return False
        self._stats[rid] = stats
        return True

    def finish(self, rid: str) -> bool:
        """End ``rid``'s root span (the request's terminal output or abort
        reached the Router); returns whether one was open."""
        stats = self._stats.pop(rid, None)
        if stats is None:
            return False
        try:
            stats.set_finished_time()
        except Exception:  # the output path must never raise
            logger.exception("request trace: could not finish the trace of %s", rid)
        return True

    def finish_batch(self, rids: Iterable[str], finished: Iterable[Any]) -> int:
        """End the root spans of the requests a scheduler output finished
        (``finished_reasons`` set per rid); returns how many."""
        return sum(self.finish(rid) for rid, reason in zip(rids, finished, strict=False) if reason)
