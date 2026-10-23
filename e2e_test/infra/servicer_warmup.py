"""The servicer's planned warm-up, read back from a worker's log.

Every gRPC servicer runs a planned warm-up before it reports SERVING
(``grpc_servicer/smg_grpc_servicer/warmup.py``), at the level and under the
budget its knob, ``SMG_<ENGINE>_SERVICER_WARMUP``, names. The pass announces
itself and its end with one log line each. This module reads those lines
back and follows a spawned worker through the pass to SERVING, so a case can
tell at which level a worker warmed up, whether the pass ran to its end
inside its budget, and whether the health flipped only after it.
"""

from __future__ import annotations

import dataclasses
import re
import time
from collections.abc import Callable

from .constants import HEALTH_CHECK_INTERVAL

#: The knob each engine's servicer reads (``grpc_servicer/smg_grpc_servicer/<engine>/warmup.py``).
SERVICER_WARMUP_ENV = {
    "vllm": "SMG_VLLM_SERVICER_WARMUP",
    "sglang": "SMG_SGLANG_SERVICER_WARMUP",
    "tokenspeed": "SMG_TOKENSPEED_SERVICER_WARMUP",
}

#: The level the e2e legs run (the job's environment) and the shipped default.
LEVEL_FAST = "fast"
LEVEL_FULL = "full"

#: The engine arguments the worker a case follows needs so the servicer's lines
#: reach its log. The harness starts the SGLang and TokenSpeed engines at their
#: ``warning`` level (``worker.py``), and both engines apply that level to the
#: process's root logger: every INFO line of the servicer package, the pass's
#: two lines among them, goes unwritten, under the Python and the Rust servicer
#: alike. The followed worker runs its engine at ``info`` instead (the last
#: ``--log-level`` on the command line wins). vLLM's servicer logs through
#: vLLM's own logger, which the harness leaves at its default, INFO.
SERVICER_LOG_ENGINE_ARGS = {
    "vllm": [],
    "sglang": ["--log-level", "info"],
    "tokenspeed": ["--log-level", "info"],
}

# "Planned warm-up starts (level full: 215 steps (1 decode staircase, 192 prefill
# shapes, 22 measured again), 1152519 prompt tokens and 526038 output tokens over
# a window of 1024 and a step of 16384; budget 300 s; <source>): readiness is held
# until it ran"
_STARTS = re.compile(
    r"Planned warm-up starts \(level (?P<level>[a-z]+): (?P<steps>\d+) steps.*?"
    r"; budget (?P<budget>\d+(?:\.\d+)?) s"
)
# "Planned warm-up over: 215 of 215 steps covered in 72.6 s (0 failed requests)"
# followed by "; the budget ended the pass before <step>", "; stopped: <why>" or
# "; the engine's cache was not flushed (...)" when that happened. The step the
# budget stopped before is the pass's next one, "prefill 37 tokens (chunk
# remainder)" when only the chunk remainders were left.
_OVER = re.compile(
    r"Planned warm-up over: (?P<covered>\d+) of (?P<planned>\d+) steps covered in "
    r"(?P<secs>\d+(?:\.\d+)?) s \((?P<failed>\d+) failed requests?\)(?P<rest>[^\r\n]*)"
)
_BUDGET_ENDED = re.compile(r"; the budget ended the pass(?: before (?P<left>[^;\r\n]*))?")
# The servicer ran no pass: the knob is off, the engine or model takes none,
# or the configuration left nothing to plan (``plan_for``'s three lines).
_NONE = re.compile(
    r"Planned warm-up(?: off \([^)\r\n]*\)|: none for this engine or model"
    r"|: nothing to plan[^\r\n]*)"
)


@dataclasses.dataclass(frozen=True)
class WarmupStart:
    """The pass as it announced itself."""

    level: str
    steps: int
    budget_secs: float


@dataclasses.dataclass(frozen=True)
class WarmupEnd:
    """The pass as it ended."""

    covered: int
    planned: int
    secs: float
    failed_requests: int
    budget_hit: bool
    stopped: bool
    #: The step the budget stopped before ("" when it did not, or stopped before none).
    left: str = ""

    @property
    def complete(self) -> bool:
        """Every planned step covered; neither the budget nor a failure streak ended it."""
        return self.covered == self.planned and not self.budget_hit and not self.stopped

    @property
    def budget_ended_in_the_remainders(self) -> bool:
        """The budget ended the pass with only chunk remainders left: every step before
        them ran (the plan puts them last), and they are its best effort under the budget."""
        return self.budget_hit and not self.stopped and self.left.endswith("(chunk remainder)")


def warmup_start(log: str) -> WarmupStart | None:
    """The start line's level, step count and budget, or ``None`` before it is logged."""
    match = _STARTS.search(log)
    if match is None:
        return None
    return WarmupStart(match["level"], int(match["steps"]), float(match["budget"]))


def warmup_end(log: str) -> WarmupEnd | None:
    """The end line's coverage and timing, or ``None`` before it is logged."""
    match = _OVER.search(log)
    if match is None:
        return None
    rest = match["rest"]
    ended = _BUDGET_ENDED.search(rest)
    return WarmupEnd(
        covered=int(match["covered"]),
        planned=int(match["planned"]),
        secs=float(match["secs"]),
        failed_requests=int(match["failed"]),
        budget_hit=ended is not None,
        stopped="; stopped:" in rest,
        left=(ended["left"] or "") if ended is not None else "",
    )


def warmup_skipped(log: str) -> str | None:
    """The servicer's line saying it ran no pass, or ``None``."""
    match = _NONE.search(log)
    return match.group(0) if match else None


@dataclasses.dataclass(frozen=True)
class WarmupOutcome:
    """A worker's pass from spawn to SERVING."""

    start: WarmupStart
    end: WarmupEnd
    #: From the watch's start (the spawn) to the first SERVING answer.
    serving_after_secs: float
    #: From the first SERVING answer to the end line's appearance in the log
    #: (0 when it was there already; the two race by milliseconds otherwise).
    end_lag_secs: float

    def describe(self) -> str:
        budget = f"under a {self.start.budget_secs:g} s budget"
        if self.end.complete:
            ended = "ran to its end"
        elif self.end.budget_ended_in_the_remainders:
            ended = f"ended on the budget in the chunk remainders (before {self.end.left})"
        elif self.end.budget_hit or self.end.stopped:
            ended = (
                "ended early"
                + (" on the budget" if self.end.budget_hit else "")
                + (" on a failure streak" if self.end.stopped else "")
            )
        else:
            # The servicer counts a step as covered only when none of its requests
            # failed: every step ran, and the uncovered ones had a failed request.
            ended = f"ran every step, {self.end.planned - self.end.covered} with failed requests"
        return (
            f"level {self.start.level}: {self.end.covered} of {self.end.planned} steps in "
            f"{self.end.secs:.1f} s {budget}, {self.end.failed_requests} failed requests, "
            f"{ended}; SERVING {self.serving_after_secs:.1f} s after the spawn, "
            f"{self.end_lag_secs:.1f} s after the end line"
        )


def _tail(log: str, lines: int = 20) -> str:
    return "\n".join(log.splitlines()[-lines:])


def watch_warmup(
    worker,
    *,
    engine_start_secs: float,
    grace_secs: float = 30.0,
    end_grace_secs: float = 5.0,
    poll_secs: float = HEALTH_CHECK_INTERVAL,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WarmupOutcome:
    """Follow ``worker`` (spawned, not waited on) through its planned warm-up to
    SERVING.

    Two bounds: ``engine_start_secs`` until the pass announces itself (the
    engine's own start), then the pass's own budget plus ``grace_secs`` from
    that moment (the end of the pass, the engine's cache flush, the health
    flip). Once SERVING is read, the end line must be in the log within
    ``end_grace_secs``: a servicer that serves before its pass is over is the
    defect this watch exists to catch. Raises ``TimeoutError`` when a bound
    passes, ``RuntimeError`` when the worker dies, ran no pass, or served
    before the pass ended.
    """
    started = clock()
    deadline = started + engine_start_secs
    start: WarmupStart | None = None
    while True:
        now = clock()
        log = worker.read_log()
        if not worker.is_alive():
            raise RuntimeError(
                f"Worker {worker.model_id} died before SERVING; its log ends:\n{_tail(log)}"
            )
        if start is None:
            start = warmup_start(log)
            if start is not None:
                deadline = max(deadline, now + start.budget_secs + grace_secs)
            else:
                reason = warmup_skipped(log)
                if reason is not None:
                    raise RuntimeError(f"the servicer of {worker.model_id} ran no pass: {reason}")
        if worker.grpc_serving():
            serving_at = clock()
            end = warmup_end(log)
            lag = 0.0
            while end is None and lag <= end_grace_secs:
                sleep(1.0)
                log = worker.read_log()
                end = warmup_end(log)
                lag = clock() - serving_at
            start = start or warmup_start(log)
            if start is None or end is None:
                missing = "start" if start is None else "end"
                raise RuntimeError(
                    f"{worker.model_id} answered SERVING with no planned warm-up {missing} line in "
                    f"its log {lag:.1f} s later (a servicer serves only after its pass); the log "
                    f"ends:\n{_tail(log)}"
                )
            return WarmupOutcome(start, end, serving_at - started, lag)
        if now >= deadline:
            phase = (
                f"its planned warm-up (budget {start.budget_secs:g} s, plus {grace_secs:g} s)"
                if start is not None
                else f"its engine's start ({engine_start_secs:g} s, no planned warm-up announced)"
            )
            raise TimeoutError(
                f"{worker.model_id} was not SERVING at the end of {phase}; the log ends:\n"
                f"{_tail(log)}"
            )
        sleep(poll_secs)
