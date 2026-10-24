"""Prometheus metrics of a Python servicer, whichever engine it serves: the
``smg_servicer_*`` families the Rust servicer exposes (the same names,
labels, bucket edges and meaning; ``crates/engine_servicer/src/metrics.rs``
is the contract, and ``tests/test_metrics.py`` reads that file to hold this
module to it), observed at the points the request path already passes, and
the ``GET /metrics`` listener that renders them.

Standard library only, on purpose: the package is installed beside every
engine, and the exposition format is small. Counters are plain integers,
histograms are bucket counts, so the registry's memory is fixed at start and
a scraper that never comes cannot grow the process (the Rust servicer's
pressure sampler has nothing to drop here: ``smg_servicer_metrics_dropped_total``
stays 0 by construction).

What the request path does, and nothing else: without an endpoint, one no-op
call per event (:data:`NO_METRICS`); with one, one clock read at arrival when
the path has no stamp of its own (every servicer hands over the one it reads
already), one clock read per output chunk, a handful of integer adds, and the
latency observations when a stream ends. The clock is the servicer's: a
servicer whose arrival stamp is the wall clock measures every interval on
it, so no second clock is read.

The listener runs on a thread of its own (the standard library's HTTP
server), binds before the engine starts, so a scrape during the engine's
load answers ``smg_servicer_state{state="starting"} 1``, and closes with the
servicer. A scrape that fails answers 500 for that scrape alone and is
counted; the next one renders again.
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socket import AF_INET6
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# The exposition content type, as the Rust servicer and the gateway answer it.
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
# How long a stop waits for the listener's thread.
STOP_SECS = 2.0
# The token-count buckets' ceiling when the model's context length is not known.
DEFAULT_MAX_MODEL_LEN = 32_768

# vLLM's time-to-first-token buckets (seconds): the same edges as the Rust
# servicer and the engine's own server, so a dashboard reads this family by `le`.
TTFT_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    640.0,
    2560.0,
)
# vLLM's inter-token latency buckets (seconds), for the per-chunk latency and
# the per-request mean.
INTER_TOKEN_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
)
# vLLM's request latency buckets (seconds), for the end-to-end latency.
REQUEST_LATENCY_BUCKETS = (
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
    1920.0,
    7680.0,
)
# Buckets of the `n` parameter.
PARAMS_N_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0)


def one_two_five_buckets(max_value: int) -> tuple[float, ...]:
    """A 1-2-5 series of bucket edges up to ``max_value`` (vLLM's
    ``build_1_2_5_buckets``): the token-count histograms' edges."""
    ceiling = float(max(int(max_value), 1))
    buckets: list[float] = []
    exponent = 0
    while True:
        for mantissa in (1.0, 2.0, 5.0):
            value = mantissa * 10.0**exponent
            if value > ceiling:
                return tuple(buckets)
            buckets.append(value)
        exponent += 1


# How a request ended, as `smg_servicer_requests_total` splits it: the
# engine's stop (EOS, a stop id or string), the token budget, the caller's
# abort or the engine's own, the engine or the wire failing the request, a
# shutdown cutting it while it was still open.
FINISH_REASONS = ("stop", "length", "abort", "error", "cut")
# Why a request was refused before it reached the engine
# (`smg_servicer_requests_refused_total`). A Python servicer refuses nothing
# itself (its engine does, after admission), so it reports the two it can
# tell before submitting: an invalid request and media it could not process;
# a duplicate id and a drain stay at 0.
REFUSALS = ("duplicate_id", "draining", "invalid_argument", "media")
# The lifecycle state a scrape reports as `smg_servicer_state`, one-hot.
STATES = ("starting", "serving", "draining", "engine_down", "warming_up")
STARTING, SERVING, DRAINING, ENGINE_DOWN, WARMING_UP = STATES

# The family names, each with the prefix; the keys are the Rust module's constants.
NAMES = {
    "REQUESTS_TOTAL": "smg_servicer_requests_total",
    "REQUESTS_REFUSED_TOTAL": "smg_servicer_requests_refused_total",
    "REQUESTS_ACTIVE": "smg_servicer_requests_active",
    "REQUEST_ERRORS_TOTAL": "smg_servicer_request_errors_total",
    "PROMPT_TOKENS_TOTAL": "smg_servicer_prompt_tokens_total",
    "PROMPT_TOKENS_CACHED_TOTAL": "smg_servicer_prompt_tokens_cached_total",
    "GENERATION_TOKENS_TOTAL": "smg_servicer_generation_tokens_total",
    "STRING_STOPS_TOTAL": "smg_servicer_string_stops_total",
    "TIME_TO_FIRST_TOKEN_SECONDS": "smg_servicer_time_to_first_token_seconds",
    "E2E_REQUEST_LATENCY_SECONDS": "smg_servicer_e2e_request_latency_seconds",
    "REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS": "smg_servicer_request_time_per_output_token_seconds",
    "INTER_TOKEN_LATENCY_SECONDS": "smg_servicer_inter_token_latency_seconds",
    "REQUEST_PROMPT_TOKENS": "smg_servicer_request_prompt_tokens",
    "REQUEST_GENERATION_TOKENS": "smg_servicer_request_generation_tokens",
    "REQUEST_PARAMS_MAX_TOKENS": "smg_servicer_request_params_max_tokens",
    "REQUEST_PARAMS_N": "smg_servicer_request_params_n",
    "STATE": "smg_servicer_state",
    "START_TIME_SECONDS": "smg_servicer_start_time_seconds",
    "METRICS_DROPPED_TOTAL": "smg_servicer_metrics_dropped_total",
    "METRICS_SCRAPE_SECONDS": "smg_servicer_metrics_scrape_seconds",
    "METRICS_SCRAPES_TOTAL": "smg_servicer_metrics_scrapes_total",
}

# The HELP line of every family, as the Rust servicer describes it.
HELP = {
    "REQUESTS_TOTAL": (
        "Requests whose stream ended, by how they ended; a fan-out (n > 1) is one request with "
        "the reason of its last choice to end, an abort winning."
    ),
    "REQUESTS_REFUSED_TOTAL": "Requests refused before they reached the engine, by reason.",
    "REQUESTS_ACTIVE": "Admitted requests still preprocessing, streaming, or unwinding an abort.",
    "REQUEST_ERRORS_TOTAL": "Requests that failed with a gRPC status, by code.",
    "PROMPT_TOKENS_TOTAL": (
        "Prompt tokens as the engine reported them at each request's first output."
    ),
    "PROMPT_TOKENS_CACHED_TOTAL": "Prompt tokens the engine served from its prefix cache.",
    "GENERATION_TOKENS_TOTAL": "Generated tokens streamed to callers.",
    "STRING_STOPS_TOTAL": "Requests this servicer ended on a stop string it matched itself.",
    "TIME_TO_FIRST_TOKEN_SECONDS": (
        "Seconds from a request's arrival at the servicer to its first output."
    ),
    "E2E_REQUEST_LATENCY_SECONDS": (
        "Seconds from a request's arrival at the servicer to the end of its stream."
    ),
    "REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS": (
        "Mean seconds per generated token after the first, per request (servicer clock)."
    ),
    "INTER_TOKEN_LATENCY_SECONDS": (
        "Seconds between consecutive output chunks of a choice, per token when a chunk carries "
        "several (servicer clock; sampled above a pending-observation bound)."
    ),
    "REQUEST_PROMPT_TOKENS": "Prompt tokens per request.",
    "REQUEST_GENERATION_TOKENS": "Generated tokens per request.",
    "REQUEST_PARAMS_MAX_TOKENS": "The max_tokens parameter of requests that set it.",
    "REQUEST_PARAMS_N": "The n parameter of requests.",
    "STATE": (
        "The servicer's lifecycle state, one-hot: starting, serving, warming_up (readiness held "
        "with admission open), draining (admission closed) or engine_down."
    ),
    "START_TIME_SECONDS": "When the servicer started, in seconds since the Unix epoch.",
    "METRICS_DROPPED_TOTAL": "Observations the pressure sampler dropped, by family.",
    "METRICS_SCRAPE_SECONDS": "How long the last scrape took to render.",
    "METRICS_SCRAPES_TOTAL": "Scrapes answered, by HTTP status code.",
}

# The gRPC codes as the `code` label spells them (the Rust servicer's
# spelling: the status code's name, lower case).
_CODES = (
    "ok",
    "cancelled",
    "unknown",
    "invalid_argument",
    "deadline_exceeded",
    "not_found",
    "already_exists",
    "permission_denied",
    "resource_exhausted",
    "failed_precondition",
    "aborted",
    "out_of_range",
    "unimplemented",
    "internal",
    "unavailable",
    "data_loss",
    "unauthenticated",
)


def code_label(code: Any) -> str:
    """The ``code`` label of a gRPC status code: ``grpc.StatusCode.INTERNAL``,
    its ``(13, "internal")`` value, its number or its name, as ``internal``."""
    name = getattr(code, "name", None)
    if isinstance(name, str):
        return name.lower()
    if isinstance(code, int) and 0 <= code < len(_CODES):
        return _CODES[code]
    if isinstance(code, tuple) and len(code) == 2 and isinstance(code[1], str):
        return code[1].lower()
    return str(code).lower()


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: float) -> str:
    """A sample value or a bucket edge as the Rust exporter prints it: an
    integral value without a fraction, the rest in the shortest form."""
    if isinstance(value, int):
        return str(value)
    if value == value and value not in (float("inf"), float("-inf")) and value == int(value):
        return str(int(value))
    return repr(value)


class _Histogram:
    """Cumulative bucket counts over fixed edges, a sum and a count."""

    __slots__ = ("count", "counts", "edges", "total")

    def __init__(self, edges: tuple[float, ...]) -> None:
        self.edges = edges
        self.counts = [0] * (len(edges) + 1)  # the last slot is +Inf
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        if value < 0.0:
            value = 0.0
        self.counts[bisect.bisect_left(self.edges, value)] += 1
        self.total += value
        self.count += 1

    def lines(self, name: str, labels: str, out: list[str]) -> None:
        cumulative = 0
        for edge, count in zip(self.edges, self.counts):
            cumulative += count
            out.append(f'{name}_bucket{{{labels},le="{_number(edge)}"}} {cumulative}')
        out.append(f'{name}_bucket{{{labels},le="+Inf"}} {self.count}')
        out.append(f"{name}_sum{{{labels}}} {_number(self.total)}")
        out.append(f"{name}_count{{{labels}}} {self.count}")


class _RankSeries:
    """The series of one data-parallel rank."""

    def __init__(self, token_buckets: tuple[float, ...], string_stops: bool) -> None:
        self.requests_total = [0] * len(FINISH_REASONS)
        self.prompt_tokens = 0
        self.prompt_tokens_cached = 0
        self.generation_tokens = 0
        # ``None`` on a servicer whose engine matches string stops (the
        # family is registered only where the servicer matches them itself).
        self.string_stops: int | None = 0 if string_stops else None
        self.time_to_first_token = _Histogram(TTFT_BUCKETS)
        self.e2e_latency = _Histogram(REQUEST_LATENCY_BUCKETS)
        self.time_per_output_token = _Histogram(INTER_TOKEN_BUCKETS)
        self.inter_token_latency = _Histogram(INTER_TOKEN_BUCKETS)
        self.request_prompt_tokens = _Histogram(token_buckets)
        self.request_generation_tokens = _Histogram(token_buckets)
        self.request_params_max_tokens = _Histogram(token_buckets)
        self.request_params_n = _Histogram(PARAMS_N_BUCKETS)


class ServicerMetrics:
    """A servicer's metrics: the registry the request path adds to and the
    scrape renders. Built with the servicer's identity (``model`` and
    ``wire``, the labels on every series) when the endpoint binds; the
    per-rank families follow once the engine's facts are known
    (:meth:`engine_facts`), which is before the first request on every
    launcher. ``clock`` is the one the servicer's request path reads: the
    arrival stamps handed to :meth:`request` are on it, and so is every
    stamp this registry reads itself."""

    def __init__(
        self, model: str, wire: str, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.model = str(model)
        self.wire = str(wire)
        self.clock = clock
        self.draining = False
        self._lock = threading.Lock()
        self._base = f'model="{_escape(self.model)}",wire="{_escape(self.wire)}"'
        self._token_buckets = one_two_five_buckets(DEFAULT_MAX_MODEL_LEN)
        self._string_stops = False
        self._ranks: list[_RankSeries] = []
        # The series of a rank the launcher did not announce, or of a request
        # whose rank this path cannot see (``rank="other"``), built on first use.
        self._other: _RankSeries | None = None
        self.active = 0
        self._refused = [0] * len(REFUSALS)
        self._errors: dict[str, int] = {}
        self._start_time = time.time()
        self._scrape_seconds = 0.0
        self._scrapes = {"200": 0, "500": 0}

    def engine_facts(
        self, *, ranks: int = 1, max_model_len: int = 0, matches_string_stops: bool = False
    ) -> None:
        """The engine's facts, once the lifecycle owner has them: the
        data-parallel width (one series set per rank, ``rank="0"`` on), the
        context length (the token-count histograms' ceiling; 0 takes a
        default), and whether this servicer matches string stops itself
        (the string-stop family is registered only then)."""
        with self._lock:
            self._token_buckets = one_two_five_buckets(
                int(max_model_len) if max_model_len else DEFAULT_MAX_MODEL_LEN
            )
            self._string_stops = bool(matches_string_stops)
            self._ranks = [
                _RankSeries(self._token_buckets, self._string_stops)
                for _ in range(max(int(ranks), 1))
            ]
            self._other = None

    def _series(self, rank: int | None) -> _RankSeries:
        """The series of ``rank``: its own within the configured width;
        ``None`` on a one-rank servicer is that rank, else, as a rank beyond
        the width, the catch-all ``other``."""
        if not self._ranks:
            self.engine_facts()
        if rank is None and len(self._ranks) == 1:
            rank = 0
        if rank is not None and 0 <= rank < len(self._ranks):
            return self._ranks[rank]
        if self._other is None:
            self._other = _RankSeries(self._token_buckets, self._string_stops)
        return self._other

    def request(
        self,
        rank: int | None = None,
        prompt_tokens: int = 0,
        max_tokens: int | None = None,
        n: int = 1,
        arrival: float | None = None,
    ) -> RequestMetrics:
        """A request was admitted and handed to the engine: the observation
        its outputs record into, ended by :meth:`RequestMetrics.end`.
        ``arrival`` is the stamp the request path read when the request
        came in (on this registry's clock); ``None`` reads one now.
        ``prompt_tokens`` is the servicer's own count (the engine's report
        replaces it at the first output)."""
        series = self._series(rank)
        with self._lock:
            self.active += 1
            if max_tokens is not None:
                series.request_params_max_tokens.observe(float(max_tokens))
            series.request_params_n.observe(float(max(int(n), 1)))
        return RequestMetrics(
            self, series, self.clock() if arrival is None else arrival, int(prompt_tokens)
        )

    def request_not_started(self, code: Any, *, refusal: str | None = None) -> None:
        """A request ended before it reached the engine: a refusal of the
        servicer's own under its ``refusal`` (one of :data:`REFUSALS`), else
        an error under its gRPC ``code``."""
        with self._lock:
            if refusal in REFUSALS:
                self._refused[REFUSALS.index(refusal)] += 1
            else:
                label = code_label(code)
                self._errors[label] = self._errors.get(label, 0) + 1

    def failed(
        self, observed: RequestMetrics | None, code: Any, *, refusal: str | None = None
    ) -> None:
        """A request failed with a gRPC status: before its submission
        (``observed`` is ``None``) it never reached the engine
        (:meth:`request_not_started`, under ``refusal`` where the servicer
        refused it itself); after, its stream failed
        (:meth:`RequestMetrics.failed`)."""
        if observed is None:
            self.request_not_started(code, refusal=refusal)
        else:
            observed.failed(code)

    def _error(self, code: Any) -> None:
        with self._lock:
            label = code_label(code)
            self._errors[label] = self._errors.get(label, 0) + 1

    def _ended(self, request: RequestMetrics, reason: int, now: float) -> None:
        series = request.series
        with self._lock:
            self.active -= 1
            series.requests_total[reason] += 1
            series.e2e_latency.observe(now - request.arrival)
            generated = request.generation_tokens
            first = request.first_output
            if generated > 1 and first is not None and request.last_output is not None:
                series.time_per_output_token.observe(
                    (request.last_output - first) / (generated - 1)
                )
            series.request_prompt_tokens.observe(float(request.prompt_tokens))
            series.request_generation_tokens.observe(float(generated))

    def render(self, state: str | None) -> str:
        """The exposition: every family, the state one-hot on ``state``."""
        with self._lock:
            return self._render(state)

    def _render(self, state: str | None) -> str:
        base = self._base
        out: list[str] = []

        def head(key: str, kind: str) -> str:
            name = NAMES[key]
            out.append(f"# HELP {name} {HELP[key]}")
            out.append(f"# TYPE {name} {kind}")
            return name

        ranks: list[tuple[str, _RankSeries]] = [
            (str(index), series) for index, series in enumerate(self._ranks)
        ]
        if self._other is not None:
            ranks.append(("other", self._other))

        def per_rank(key: str, kind: str, value: Callable[[_RankSeries], Any]) -> None:
            if not ranks:
                return
            name = head(key, kind)
            for rank, series in ranks:
                labels = f'{base},rank="{rank}"'
                sample = value(series)
                if isinstance(sample, _Histogram):
                    sample.lines(name, labels, out)
                else:
                    out.append(f"{name}{{{labels}}} {_number(sample)}")

        if ranks:
            name = head("REQUESTS_TOTAL", "counter")
            for rank, series in ranks:
                for reason, count in zip(FINISH_REASONS, series.requests_total):
                    out.append(f'{name}{{{base},rank="{rank}",finish_reason="{reason}"}} {count}')
        name = head("REQUESTS_REFUSED_TOTAL", "counter")
        for reason, count in zip(REFUSALS, self._refused):
            out.append(f'{name}{{{base},reason="{reason}"}} {count}')
        name = head("REQUESTS_ACTIVE", "gauge")
        out.append(f"{name}{{{base}}} {self.active}")
        if self._errors:
            name = head("REQUEST_ERRORS_TOTAL", "counter")
            for code, count in self._errors.items():
                out.append(f'{name}{{{base},code="{code}"}} {count}')
        per_rank("PROMPT_TOKENS_TOTAL", "counter", lambda s: s.prompt_tokens)
        per_rank("PROMPT_TOKENS_CACHED_TOTAL", "counter", lambda s: s.prompt_tokens_cached)
        per_rank("GENERATION_TOKENS_TOTAL", "counter", lambda s: s.generation_tokens)
        if self._string_stops:
            per_rank("STRING_STOPS_TOTAL", "counter", lambda s: s.string_stops or 0)
        per_rank("TIME_TO_FIRST_TOKEN_SECONDS", "histogram", lambda s: s.time_to_first_token)
        per_rank("E2E_REQUEST_LATENCY_SECONDS", "histogram", lambda s: s.e2e_latency)
        per_rank(
            "REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS", "histogram", lambda s: s.time_per_output_token
        )
        per_rank("INTER_TOKEN_LATENCY_SECONDS", "histogram", lambda s: s.inter_token_latency)
        per_rank("REQUEST_PROMPT_TOKENS", "histogram", lambda s: s.request_prompt_tokens)
        per_rank("REQUEST_GENERATION_TOKENS", "histogram", lambda s: s.request_generation_tokens)
        per_rank("REQUEST_PARAMS_MAX_TOKENS", "histogram", lambda s: s.request_params_max_tokens)
        per_rank("REQUEST_PARAMS_N", "histogram", lambda s: s.request_params_n)
        name = head("STATE", "gauge")
        for candidate in STATES:
            out.append(f'{name}{{{base},state="{candidate}"}} {int(candidate == state)}')
        name = head("START_TIME_SECONDS", "gauge")
        out.append(f"{name}{{{base}}} {_number(self._start_time)}")
        name = head("METRICS_DROPPED_TOTAL", "counter")
        out.append(f'{name}{{{base},family="inter_token_latency"}} 0')
        name = head("METRICS_SCRAPE_SECONDS", "gauge")
        out.append(f"{name}{{{base}}} {_number(self._scrape_seconds)}")
        name = head("METRICS_SCRAPES_TOTAL", "counter")
        for code, count in self._scrapes.items():
            out.append(f'{name}{{{base},code="{code}"}} {count}')
        out.append("")
        return "\n".join(out)

    def scrape(self, state: str | None | Callable[[], str | None]) -> tuple[int, bytes]:
        """The scrape as the listener serves it: the status and the body; a
        failure inside the render is one 500, counted, and the next scrape
        renders again. A lifecycle probe handed in as ``state`` is read under
        the same guard: a probe that raises (an engine object read from the
        listener thread) is a counted 500 too, not a closed connection."""
        started = time.perf_counter()
        try:
            body = self.render(state() if callable(state) else state)
        except Exception:
            logger.exception("the metrics scrape failed; the next one renders again")
            with self._lock:
                self._scrapes["500"] += 1
            return 500, b"metrics scrape failed; the next scrape renders again\n"
        with self._lock:
            self._scrape_seconds = time.perf_counter() - started
            self._scrapes["200"] += 1
        return 200, body.encode("utf-8")


def rank_among(managed: Sequence[int], pinned: int | None) -> int | None:
    """The series a request records on, by the place of its pin among the
    ranks the servicer serves (``managed``, in order; the series are labelled
    by that place, as the Rust servicer labels its engines by index):
    ``None`` for a request that names none (the one rank when the servicer
    serves one, else ``other``), a place beyond the width for a pin outside
    the managed ranks."""
    if pinned is None:
        return None
    return managed.index(pinned) if pinned in managed else len(managed)


class RequestMetrics:
    """One request's observation, shared by its choices (a fan-out is one
    request); the request-level families are recorded by :meth:`end`."""

    __slots__ = (
        "_aborted",
        "_errored",
        "_last_chunk",
        "_last_finish",
        "arrival",
        "first_output",
        "generation_tokens",
        "last_output",
        "metrics",
        "prompt_counted",
        "prompt_tokens",
        "series",
    )

    def __init__(
        self, metrics: ServicerMetrics, series: _RankSeries, arrival: float, prompt_tokens: int
    ) -> None:
        self.metrics = metrics
        self.series = series
        self.arrival = arrival
        self.first_output: float | None = None
        self.last_output: float | None = None
        self.prompt_counted = False
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = 0
        # The reason of the last choice to end (an index into FINISH_REASONS).
        self._last_finish: int | None = None
        self._aborted = False
        self._errored = False
        # The last chunk's stamp per choice, for the inter-token latency.
        self._last_chunk: dict[int, float] = {}

    def chunk(
        self, tokens: int, prompt_tokens: int = 0, cached_tokens: int = 0, index: int = 0
    ) -> None:
        """An output chunk of ``tokens`` new tokens of choice ``index``
        reached the servicer, with the engine's prompt figures on it (the
        first non-zero report counts once per request). One clock read and
        one lock acquisition per chunk; the scrape reads under the same lock,
        so a histogram never renders half-updated."""
        metrics = self.metrics
        now = metrics.clock()
        series = self.series
        with metrics._lock:
            if prompt_tokens > 0 and not self.prompt_counted:
                self.prompt_counted = True
                series.prompt_tokens += prompt_tokens
                series.prompt_tokens_cached += min(cached_tokens, prompt_tokens)
                self.prompt_tokens = prompt_tokens
            if self.first_output is None:
                self.first_output = now
                series.time_to_first_token.observe(now - self.arrival)
            if tokens > 0:
                series.generation_tokens += tokens
                self.generation_tokens += tokens
                last = self._last_chunk.get(index)
                if last is not None:
                    series.inter_token_latency.observe((now - last) / tokens)
                self._last_chunk[index] = now
                self.last_output = now

    def complete(
        self,
        finish_reason: str,
        prompt_tokens: int = 0,
        cached_tokens: int = 0,
        index: int = 0,
        *,
        tokens: int = 0,
    ) -> None:
        """The engine's terminal for choice ``index``, with the reason it
        carries (``stop``, ``length``, ``abort``, ``error``; anything else
        counts as a stop) and, on a wire whose terminal carries the output
        of a non-streamed request, its ``tokens`` (counted as a chunk). A
        terminal after the choice's chunks has nothing left to observe: it
        records the reason without a clock read or the lock."""
        if (
            tokens > 0
            or self.first_output is None
            or (prompt_tokens > 0 and not self.prompt_counted)
        ):
            self.chunk(tokens, prompt_tokens, cached_tokens, index)
        self.ended(finish_reason)

    def chunk_of(self, response: Any) -> None:
        """:meth:`chunk` from a response whose ``chunk`` is set, as every
        wire shapes it (``token_ids``, ``prompt_tokens``, ``cached_tokens``,
        ``index``). The message is read here, so a servicer without an
        endpoint (:data:`NO_METRICS`) reads nothing of it."""
        chunk = response.chunk
        self.chunk(len(chunk.token_ids), chunk.prompt_tokens, chunk.cached_tokens, chunk.index)

    def complete_of(self, response: Any, *, with_output: bool = False) -> None:
        """:meth:`complete` from a response whose ``complete`` is set, as
        every wire shapes it; ``with_output`` where the terminal carries the
        whole output (a non-streamed request), whose ``completion_tokens``
        count as one chunk."""
        complete = response.complete
        self.complete(
            complete.finish_reason,
            complete.prompt_tokens,
            complete.cached_tokens,
            complete.index,
            tokens=complete.completion_tokens if with_output else 0,
        )

    def ended(self, finish_reason: str) -> None:
        """A choice ended with ``finish_reason`` without a terminal of the
        engine's (a completion this servicer synthesized)."""
        if finish_reason == "abort":
            self._aborted = True
            self._last_finish = 2
        elif finish_reason == "length":
            self._last_finish = 1
        elif finish_reason == "error":
            self._last_finish = 3
        else:
            self._last_finish = 0

    def string_stop(self) -> None:
        """This servicer ended a choice on a stop string it matched."""
        if self.series.string_stops is not None:
            self.series.string_stops += 1
        self.ended("stop")

    def failed(self, code: Any) -> None:
        """The stream ended with a gRPC status."""
        if not self._errored:
            self._errored = True
            self.metrics._error(code)
        self._last_finish = 3

    def end(self) -> None:
        """The stream is over, however it ended: the request's end. A
        request no choice of which ended was left by its caller with the
        stream open, which aborts the engine side, so it counts as an
        abort; one left open while the servicer drains was cut."""
        if self._aborted:
            reason = 2
        elif self._last_finish is None:
            reason = 4 if self.metrics.draining else 2
        else:
            reason = self._last_finish
        self.metrics._ended(self, reason, self.metrics.clock())


class _NoMetrics:
    """What a servicer without an endpoint holds: every call is a no-op."""

    __slots__ = ()

    def request(self, *args: Any, **kwargs: Any) -> _NoMetrics:
        return self

    def request_not_started(self, *args: Any, **kwargs: Any) -> None:
        return None

    def engine_facts(self, *args: Any, **kwargs: Any) -> None:
        return None

    def chunk(self, *args: Any, **kwargs: Any) -> None:
        return None

    def chunk_of(self, *args: Any, **kwargs: Any) -> None:
        return None

    def complete(self, *args: Any, **kwargs: Any) -> None:
        return None

    def complete_of(self, *args: Any, **kwargs: Any) -> None:
        return None

    def ended(self, *args: Any, **kwargs: Any) -> None:
        return None

    def string_stop(self) -> None:
        return None

    def failed(self, *args: Any, **kwargs: Any) -> None:
        return None

    def end(self) -> None:
        return None


#: The registry of a servicer that exposes no endpoint: nothing is recorded.
NO_METRICS = _NoMetrics()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "smg-servicer-metrics"
    sys_version = ""

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/metrics":
            self._answer(404, b"not found\n")
            return
        status, body = self.server.endpoint.scrape()  # type: ignore[attr-defined]
        self._answer(status, body)

    def _answer(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", CONTENT_TYPE)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], endpoint: MetricsEndpoint) -> None:
        if ":" in address[0]:
            self.address_family = AF_INET6
        self.endpoint = endpoint
        super().__init__(address, _Handler)


def split_bind_address(bind_address: str) -> tuple[str, int]:
    """``host:port`` (an IPv6 host bracketed) as the listener binds it."""
    host, _, port = bind_address.rpartition(":")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host or "0.0.0.0", int(port)


class MetricsEndpoint:
    """The endpoint as a lifecycle owner holds it: the registry
    (:attr:`metrics`), the listener bound at :meth:`start` (before the
    engine starts: a scrape during its load reads ``starting``), and the
    servicer's lifecycle, read at every scrape from the function the owner
    supplies once the servicer exists (:meth:`serve_from`); ``draining``
    once the owner announced its shutdown (:meth:`note_draining`), whatever
    the function says."""

    def __init__(
        self,
        bind_address: str,
        *,
        model: str,
        wire: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bind_address = bind_address
        self.metrics = ServicerMetrics(model, wire, clock=clock)
        self._lifecycle: Callable[[], str] | None = None
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> MetricsEndpoint:
        """Bind the listener and serve from a thread of its own. A bind
        failure is the caller's to refuse (an ``OSError``): the operator
        asked for the port."""
        self._server = _Server(split_bind_address(self.bind_address), self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.5},
            name="smg-servicer-metrics",
            daemon=True,
        )
        self._thread.start()
        logger.info("Prometheus metrics on http://%s/metrics", self.address)
        return self

    @property
    def address(self) -> str:
        """The bound ``host:port`` (the port as bound: 0 asked for an
        ephemeral one), the bind address before the start."""
        if self._server is None:
            return self.bind_address
        host, port = self._server.server_address[:2]
        return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"

    def serve_from(self, lifecycle: Callable[[], str]) -> None:
        """The servicer exists: ``lifecycle`` says, at every scrape, whether
        it is serving, warming up or without its engine."""
        self._lifecycle = lifecycle

    def note_draining(self) -> None:
        """The lifecycle owner announced its shutdown: the state reads
        ``draining`` from here on, and a request left open counts as cut."""
        self.metrics.draining = True

    def state(self) -> str:
        if self.metrics.draining:
            return DRAINING
        if self._lifecycle is None:
            return STARTING
        state = self._lifecycle()
        return state if state in STATES else SERVING

    def scrape(self) -> tuple[int, bytes]:
        return self.metrics.scrape(self.state)

    def stop(self, timeout: float = STOP_SECS) -> None:
        """Close the listener and join its thread, within ``timeout``."""
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout)


# The endpoint of this process's launcher, for a servicer the launcher builds
# without a namespace of its own (an upstream launcher constructs the servicer
# itself and passes nothing of the launcher's flags): set by the launcher's
# switch when it binds the listener, read by the servicer's constructor.
_launcher_endpoint: MetricsEndpoint | None = None


def remember_launcher_endpoint(endpoint: MetricsEndpoint | None) -> None:
    """Keep the launcher's endpoint for :func:`launcher_endpoint`."""
    global _launcher_endpoint  # one launcher per process
    _launcher_endpoint = endpoint


def launcher_endpoint() -> MetricsEndpoint | None:
    """The endpoint this process's launcher bound, if any."""
    return _launcher_endpoint
