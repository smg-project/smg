"""The planned warm-up a servicer runs before it reports SERVING, the same
behind every engine.

An engine selects and compiles kernels lazily: the first step it runs at a
token count it has not seen (a new prefill length class, a new running-batch
size, a count that breaks an alignment its compiler specialised on) pays a
kernel search or a compile of up to several seconds, and every request in
flight during that step waits. The engine's own start-up warms a fixed set of
shapes (its graph-capture sizes, one tuning pass at its largest step), so the
shapes a traffic ramp reaches first are paid for by the ramp's requests, on
every engine of a fleet at once when one router feeds them the same traffic.

This module plans the shapes once from what the engine's configuration admits
and drives them through the servicer's own request path before the health
flips, so the first-time costs are paid while nothing is routed here:

- the **decode staircase**: as many concurrent requests as the running window
  holds, each asking for one more output token than the one before, so the
  running batch shrinks by one per step and every batch size from the window
  down to one runs once;
- the **prefill ladder**: one request per prefill length class the admission
  step admits (every :data:`PREFILL_CLASS_TOKENS`-token class up to the
  step's token budget at the ``full`` level; the engine tuners' coarser
  bucket grid at the ``fast`` level), the engine's graph-capture sizes, and
  the step itself;
- **odd variants**: lengths off the compilers' alignment (one above a
  bucket), which specialise separately;
- a **second pass** over the bucket grid, whose timings show what the first
  pass paid;
- the **chunk remainders**, when the engine module asks for them
  (:attr:`WarmupInputs.every_prefill_length`) and the level is ``full``:
  every prompt length up to the step the ladder left out, last, in an order
  whose every prefix is a uniform spread of the range (:func:`spread_order`),
  so a budget cut leaves an even coverage rather than a missing tail. A
  chunked prefill's last chunk is any count below the step, and an engine
  whose kernel choice at a prefill length comes from a library it reaches
  lazily loads a never-used kernel on that count's first use; during serving,
  with a collective of the overlapped previous chunk still in flight, that
  load has deadlocked the forward. The ``fast`` level plans none of them and
  leaves that hazard open (the start log says so): it is for a test leg or a
  quick restart, and the shipped default is ``full``.

Every request carries its own token ids (no two requests share a prefix
block), so prefix caching never turns a planned prefill into a hit, and the
engine's cache is flushed at the end. A failed request is a logged line, never
the end of the start; a time budget bounds the whole pass, each request's
timeout clamped to what is left of it. Under data parallelism every rank is a
scheduler of its own: the plan runs the staircase once per rank and every
prefill shape on every rank at once, each request pinned to its rank, where
the wire can pin one (:attr:`WarmupInputs.data_parallel_size`). The engine
modules supply the inputs (:class:`WarmupInputs`) and the wire
(:class:`WarmupWire`): nothing here knows which engine it warms.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import logging
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, Protocol

logger = logging.getLogger(__name__)

LEVEL_FAST = "fast"
LEVEL_FULL = "full"
LEVELS = (LEVEL_FAST, LEVEL_FULL)
DEFAULT_LEVEL = LEVEL_FULL
#: The whole pass is bounded by this many seconds unless the knob says
#: otherwise. An engine module whose plan is longer (the chunk remainders)
#: states a default of its own through :func:`parse_setting`'s
#: ``default_budget_secs``; the engines that plan the length classes alone
#: keep this one.
DEFAULT_BUDGET_SECS = 300.0
#: How many chunk remainders go by between the INFO lines that report their
#: progress (each remainder's own line is DEBUG: thousands of one-token
#: requests at a fraction of a second each say little one by one).
REMAINDER_PROGRESS_EVERY = 500
#: The values of the knob that run no warm-up.
OFF_VALUES = ("off", "0", "false", "no", "none")
#: The width of a prefill length class the ``full`` level covers one by one:
#: the engines' tuners key their kernel choice on the token count rounded up
#: to this grid, so one request per class visits every key the step admits.
PREFILL_CLASS_TOKENS = 128
#: The prompt of a staircase request: short, so the window's worth of prompts
#: fits one admission step and the staircase measures decode steps only.
STAIRCASE_PROMPT_TOKENS = 8
#: Token ids come from this region of the vocabulary: above the ids the
#: tokenizers keep for their control tokens, below the ids the large
#: vocabularies add last (media placeholders among them).
TOKEN_ID_FLOOR = 1000
TOKEN_ID_SPAN_MAX = 30000
#: How many whole steps may fail in a row before the pass stops: the engine
#: is not taking requests, and the start is not held for it.
CONSECUTIVE_FAILED_STEPS = 3
#: How long one warm-up request may take; a step at a new shape pays a compile.
REQUEST_TIMEOUT_SECS = 600.0
#: A request's timeout never drops below this when the budget's remainder
#: clamps it: the engine gets a second to answer the shape before the request
#: counts as failed and the pass ends on the budget.
REQUEST_TIMEOUT_FLOOR_SECS = 1.0
#: The running window the staircase assumes when the engine's configuration
#: does not state one (the engines' common default).
DEFAULT_WINDOW = 128


# ---------------------------------------------------------------------------
# The knob
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WarmupSetting:
    """What the knob asked for: whether a warm-up runs, at which level, under
    which budget in seconds."""

    enabled: bool
    level: str = DEFAULT_LEVEL
    budget_secs: float = DEFAULT_BUDGET_SECS

    @property
    def text(self) -> str:
        if not self.enabled:
            return "off"
        return f"{self.level}:{self.budget_secs:g}"


def parse_setting(
    text: str | None, *, name: str, default_budget_secs: float = DEFAULT_BUDGET_SECS
) -> WarmupSetting:
    """The setting a value of the knob ``name`` stands for: unset or empty is
    the default (``full``, ``default_budget_secs``: :data:`DEFAULT_BUDGET_SECS`
    unless the engine module states its own); one of :data:`OFF_VALUES` runs
    none; ``fast`` or ``full`` picks the level under the default budget; a
    number of seconds is the budget at the default level;
    ``<level>:<seconds>`` sets both. Anything else is a ``ValueError`` naming
    the variable and the value."""
    raw = (text or "").strip().lower()
    if not raw:
        return WarmupSetting(True, DEFAULT_LEVEL, default_budget_secs)
    if raw in OFF_VALUES:
        return WarmupSetting(False)
    level, separator, budget = raw.partition(":")
    if level in LEVELS and not separator:
        return WarmupSetting(True, level, default_budget_secs)
    if not separator and level not in LEVELS:
        level, budget = DEFAULT_LEVEL, level
    if level in LEVELS:
        try:
            secs = float(budget)
        except ValueError:
            secs = float("nan")
        if secs == secs and secs > 0 and secs != float("inf"):
            return WarmupSetting(True, level, secs)
    raise ValueError(
        f"{name}={text!r}: expected one of {', '.join(OFF_VALUES[:1] + LEVELS)}, a budget in "
        f"seconds, or <level>:<seconds> (the default is {DEFAULT_LEVEL}:{default_budget_secs:g})"
    )


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WarmupInputs:
    """What the planner needs of the engine's configuration. A zero stands
    for a figure the configuration does not carry: the window falls back to
    :data:`DEFAULT_WINDOW`, an unknown step or vocabulary plans nothing."""

    #: The running window: how many requests the scheduler runs at once.
    max_num_seqs: int
    #: The admission step: how many prompt tokens one scheduler step admits.
    max_num_batched_tokens: int
    #: The context length a request may not exceed, prompt and output together.
    max_model_len: int
    #: The vocabulary size the token ids are drawn below.
    vocab_size: int
    #: The engine's graph-capture sizes in tokens, when the configuration
    #: states them: each one is a shape of its own.
    capture_sizes: tuple[int, ...] = ()
    #: Where the figures came from, for the start line.
    source: str = ""
    #: How many positions of the context the engine's scheduler keeps back
    #: beyond a request's prompt and its output budget: a planned prompt is
    #: at most the context less the request's output less this many tokens.
    #: The engines differ (one admits a prompt and an output budget that
    #: together fill the context; the others keep one or more positions back
    #: and refuse, or clamp the output of, a request that reaches them), so
    #: an engine module states what its scheduler's own admission computes,
    #: read off the engine's configuration.
    context_reserve_tokens: int = 0
    #: How many data-parallel ranks the servicer fronts, each a scheduler of
    #: its own with the window and the step above. Above one, the plan runs
    #: every shape on every rank, each request pinned to its rank; an engine
    #: module states more than one only where its wire can pin a rank.
    data_parallel_size: int = 1
    #: Whether the plan visits every prompt length up to the step, not only
    #: the length classes and the grid: an engine module sets it when the
    #: engine's kernel choice at a prefill length comes from a library the
    #: servicer cannot bucket and a first use during serving is a hazard, so
    #: every length is used once here, before anything is routed (a chunked
    #: prefill's last chunk is any count below the step). Honoured at the
    #: ``full`` level alone: a ``fast`` start plans the classes and leaves the
    #: hazard open, and :func:`plan_for` warns so.
    every_prefill_length: bool = False
    #: The bootstrap host the engine's scheduler takes as a transfer to
    #: nobody (its own warm-up's stand-in), when the engine is one leg of a
    #: disaggregated pair: every request carries it, so the leg warms up
    #: without its peer. ``None`` for a single engine.
    lone_leg_bootstrap: str | None = None


@dataclasses.dataclass(frozen=True)
class WarmupRequest:
    request_id: str
    #: How many prompt tokens the request carries. The ids themselves are
    #: drawn when a wire reads :attr:`token_ids`: a plan over every count of
    #: a step holds thousands of requests and millions of tokens, and keeps
    #: counts, not tokens.
    prompt_tokens: int
    max_tokens: int
    #: The ordinal that seeds the request's own token ids, and the vocabulary
    #: they are drawn below (:func:`token_ids`).
    seed: int
    vocab_size: int
    #: The data-parallel rank the request is pinned to; ``None`` leaves the
    #: servicer's own routing to pick the engine (one rank, or no pinning).
    data_parallel_rank: int | None = None
    #: A lone leg's stand-in bootstrap host
    #: (:attr:`WarmupInputs.lone_leg_bootstrap`), for the wire to carry.
    lone_leg_bootstrap: str | None = None

    @property
    def token_ids(self) -> tuple[int, ...]:
        """The request's own token ids, drawn afresh on each read."""
        return token_ids(self.prompt_tokens, self.seed, self.vocab_size)


@dataclasses.dataclass(frozen=True)
class WarmupStep:
    """One scheduler shape: the requests sent together to reach it."""

    name: str
    # "decode", "prefill", "again" (the measuring second pass) or "remainder"
    # (a prompt length the ladder left out, for an engine that asks for all)
    kind: str
    requests: tuple[WarmupRequest, ...]

    @property
    def prompt_tokens(self) -> int:
        return sum(request.prompt_tokens for request in self.requests)

    @property
    def output_tokens(self) -> int:
        return sum(request.max_tokens for request in self.requests)


@dataclasses.dataclass(frozen=True)
class WarmupPlan:
    level: str
    inputs: WarmupInputs
    steps: tuple[WarmupStep, ...]

    @property
    def prompt_tokens(self) -> int:
        return sum(step.prompt_tokens for step in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(step.output_tokens for step in self.steps)

    def describe(self) -> str:
        kinds = {
            kind: sum(1 for step in self.steps if step.kind == kind)
            for kind in ("decode", "prefill", "again", "remainder")
        }
        ranks = self.inputs.data_parallel_size
        return (
            f"level {self.level}: {len(self.steps)} steps ({kinds['decode']} decode "
            f"staircase{'s' if kinds['decode'] != 1 else ''}, "
            f"{kinds['prefill']} prefill shapes, {kinds['again']} measured again"
            + (f", {kinds['remainder']} chunk remainders" if kinds["remainder"] else "")
            + f"), {self.prompt_tokens} prompt tokens and {self.output_tokens} output tokens over "
            f"a window of {self.inputs.max_num_seqs} and a step of "
            f"{self.inputs.max_num_batched_tokens}"
            + (f" on each of {ranks} data-parallel ranks" if ranks > 1 else "")
            + (
                ", the requests carrying the lone leg's stand-in bootstrap"
                if self.inputs.lone_leg_bootstrap
                else ""
            )
        )


def hybrid_buckets(limit: int) -> list[int]:
    """The engine tuners' bucket grid up to ``limit``: powers of two to 256,
    steps of 256 to 2048, steps of 512 to 4096, powers of two above; and
    ``limit`` itself when it is not on the grid."""
    if limit < 1:
        return []
    buckets: list[int] = []
    value = 1
    while value <= min(limit, 256):
        buckets.append(value)
        value *= 2
    buckets += list(range(512, min(limit, 2048) + 1, 256))
    buckets += list(range(2560, min(limit, 4096) + 1, 512))
    value = 8192
    while value <= limit:
        buckets.append(value)
        value *= 2
    if buckets[-1] != limit:
        buckets.append(limit)
    return buckets


def prefill_classes(limit: int, width: int = PREFILL_CLASS_TOKENS) -> list[int]:
    """Every ``width``-token length class up to ``limit`` (its largest member
    each), and ``limit`` itself."""
    if limit < 1:
        return []
    classes = list(range(width, limit + 1, width))
    if not classes or classes[-1] != limit:
        classes.append(limit)
    return classes


def odd_variants(counts: Iterable[int], limit: int) -> list[int]:
    """One token above each count at or above 16 (a length off the
    compilers' 16-token alignment, which they specialise separately), within
    ``limit``."""
    return [count + 1 for count in counts if count >= 16 and count + 1 <= limit]


def spread_order(count: int) -> list[int]:
    """``1..count`` in the order of the bit-reversed integers (the van der
    Corput order): the power of two at the top of the range first, then the
    halves, the quarters, and so on, so every prefix of the order is a
    near-uniform spread of the range and a pass that stops partway leaves an
    even coverage rather than a missing tail."""
    if count < 1:
        return []
    bits = count.bit_length()
    order: list[int] = []
    for index in range(1 << bits):
        value = int(f"{index:0{bits}b}"[::-1], 2)
        if 1 <= value <= count:
            order.append(value)
    return order


def token_ids(count: int, seed: int, vocab_size: int) -> tuple[int, ...]:
    """``count`` token ids for the request with ordinal ``seed``: drawn from
    the vocabulary's ordinary region, a different sequence per seed, so no two
    requests share a prefix block."""
    floor = TOKEN_ID_FLOOR if vocab_size > 2 * TOKEN_ID_FLOOR else 0
    span = min(max(vocab_size - floor, 1), TOKEN_ID_SPAN_MAX)
    draw = random.Random(seed)
    return tuple(floor + draw.randrange(span) for _ in range(count))


class _Requests:
    """The requests of one plan, numbered: the ordinal seeds the token ids."""

    def __init__(self, vocab_size: int, lone_leg_bootstrap: str | None = None) -> None:
        self._vocab_size = vocab_size
        self._lone_leg_bootstrap = lone_leg_bootstrap
        self._next = 0

    def one(self, prompt_tokens: int, max_tokens: int, rank: int | None = None) -> WarmupRequest:
        ordinal = self._next
        self._next += 1
        # The exact form the Rust servicers admit while readiness is held
        # (`is_warmup_request_id` in the engine servicer crate's request
        # registry): the prefix, the ordinal, nothing else.
        return WarmupRequest(
            request_id=f"smg-warmup-{ordinal}",
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            seed=ordinal,
            vocab_size=self._vocab_size,
            data_parallel_rank=rank,
            lone_leg_bootstrap=self._lone_leg_bootstrap,
        )


def plan_warmup(inputs: WarmupInputs, level: str = DEFAULT_LEVEL) -> WarmupPlan:
    """The steps for ``inputs`` at ``level``, in the order they run: the
    decode staircase first (cheap, and every ramp's first shapes), then the
    prefill shapes by length, the odd variants, the second pass, and, when
    the inputs ask for every prefill length, the chunk remainders in
    :func:`spread_order` at the ``full`` level (the ``fast`` level keeps the
    merged ladder: a test leg or a quick start takes the classes alone, and
    the shipped default is ``full``). A prompt never exceeds the admission
    step, nor the context less its output and the positions the engine's
    scheduler keeps back (:attr:`WarmupInputs.context_reserve_tokens`), so
    no step reaches a shape the engine refuses. Under data parallelism
    (``inputs.data_parallel_size`` above one) the staircase runs once per
    rank and every prefill shape goes to every rank at once, each request
    pinned to its rank."""
    if level not in LEVELS:
        raise ValueError(f"unknown warm-up level {level!r}; expected one of {LEVELS}")
    steps: list[WarmupStep] = []
    if inputs.max_num_batched_tokens <= 0 or inputs.vocab_size <= 0:
        return WarmupPlan(level, inputs, ())
    requests = _Requests(inputs.vocab_size, inputs.lone_leg_bootstrap)
    window = inputs.max_num_seqs if inputs.max_num_seqs > 0 else DEFAULT_WINDOW
    # The longest prompt a one-token request may carry: the context less the
    # token and the positions the engine keeps back (the step alone when the
    # configuration states no context).
    context = (
        inputs.max_model_len if inputs.max_model_len > 0 else inputs.max_num_batched_tokens + 1
    )
    reserve = max(inputs.context_reserve_tokens, 0) if inputs.max_model_len > 0 else 0
    limit = min(inputs.max_num_batched_tokens, context - 1 - reserve)
    if limit < 1:
        return WarmupPlan(level, inputs, ())
    ranks: tuple[int | None, ...] = (
        tuple(range(inputs.data_parallel_size)) if inputs.data_parallel_size > 1 else (None,)
    )

    # The decode staircase: as many requests as fit the window and one
    # admission step of short prompts, asking for 2..N+1 output tokens (the
    # first token comes out of the prefill step; the decode batch holds the
    # requests still generating after it).
    count = min(window, inputs.max_num_batched_tokens // STAIRCASE_PROMPT_TOKENS)
    count = min(count, max(context - 1 - reserve - STAIRCASE_PROMPT_TOKENS, 0))
    for rank in ranks if count >= 1 else ():
        on_rank = f" on rank {rank}" if rank is not None else ""
        steps.append(
            WarmupStep(
                name=f"decode batch {count} down to 1 ({count} requests{on_rank})",
                kind="decode",
                requests=tuple(
                    requests.one(STAIRCASE_PROMPT_TOKENS, index + 2, rank) for index in range(count)
                ),
            )
        )

    grid = hybrid_buckets(limit)
    if level == LEVEL_FULL:
        lengths = set(grid) | set(prefill_classes(limit))
    else:
        lengths = set(grid)
    lengths |= {size for size in inputs.capture_sizes if 1 <= size <= limit}
    lengths |= set(odd_variants(sorted(grid), limit))
    for length in sorted(lengths):
        steps.append(
            WarmupStep(
                name=f"prefill {length} tokens",
                kind="prefill",
                requests=tuple(requests.one(length, 1, rank) for rank in ranks),
            )
        )
    for length in grid:
        steps.append(
            WarmupStep(
                name=f"prefill {length} tokens again",
                kind="again",
                requests=tuple(requests.one(length, 1, rank) for rank in ranks),
            )
        )
    if inputs.every_prefill_length and level == LEVEL_FULL:
        for length in spread_order(limit):
            if length in lengths:
                continue
            steps.append(
                WarmupStep(
                    name=f"prefill {length} tokens (chunk remainder)",
                    kind="remainder",
                    requests=tuple(requests.one(length, 1, rank) for rank in ranks),
                )
            )
    return WarmupPlan(level, inputs, tuple(steps))


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


class WarmupWire(Protocol):
    """How a pass reaches the engine: one request to completion, the cache
    flush, the close. A synchronous wire blocks; an asynchronous one returns
    awaitables (the pass tells them apart by what ``generate`` returns)."""

    def generate(self, request: WarmupRequest, timeout: float) -> Any: ...

    def flush_cache(self) -> Any: ...

    def close(self) -> Any: ...


#: The finishes with which an engine ends a request it did not run to its
#: end: refused at admission (a prompt past its input bound, a parameter its
#: scheduler rejects) or cut short (a pause, a failure). Whatever the reason,
#: and whether or not the wire carries it, nothing of the step's shape is
#: known to be warm: the pass counts such a request as failed.
UNSERVED_FINISHES = frozenset({"abort", "error"})


def unserved(finish_reason: str | None, output_tokens: int) -> str | None:
    """Why a request that completed was not served, or ``None`` when it
    was: the finish the engine ended it with, when that finish is one of
    :data:`UNSERVED_FINISHES`, and how many tokens came before it (none for
    a refusal at admission). A wire raises these words from ``generate`` so
    the pass records the request as failed, next to a raised status, and
    the step as not covered. An empty or absent finish is not a refusal (a
    completion a stand-in answers without one)."""
    if (finish_reason or "") not in UNSERVED_FINISHES:
        return None
    produced = (
        "before any token"
        if output_tokens == 0
        else f"after {output_tokens} token{'s' if output_tokens != 1 else ''}"
    )
    return f"the engine ended the request with finish_reason {finish_reason!r} {produced}"


@dataclasses.dataclass
class StepResult:
    step: WarmupStep
    secs: float
    errors: int


@dataclasses.dataclass
class WarmupReport:
    plan: WarmupPlan
    results: list[StepResult] = dataclasses.field(default_factory=list)
    secs: float = 0.0
    budget_hit: bool = False
    stopped: str | None = None
    flushed: bool | None = None
    #: Why the flush was refused, when the wire raised or the servicer
    #: answered the refusal with a message: the servicer's own words (a
    #: status' details, else its code), for the end line.
    flush_refusal: str | None = None

    @property
    def covered(self) -> int:
        """How many steps ran without a failed request."""
        return sum(1 for result in self.results if result.errors == 0)

    @property
    def errors(self) -> int:
        return sum(result.errors for result in self.results)

    def summary(self) -> str:
        text = (
            f"{self.covered} of {len(self.plan.steps)} steps covered in {self.secs:.1f} s "
            f"({self.errors} failed requests)"
        )
        if self.budget_hit:
            left = (
                self.plan.steps[len(self.results)].name
                if len(self.results) < len(self.plan.steps)
                else ""
            )
            text += f"; the budget ended the pass{f' before {left}' if left else ''}"
        elif self.stopped:
            text += f"; stopped: {self.stopped}"
        if self.flushed is False:
            text += "; the engine's cache was not flushed (the warm-up's blocks stay cached)"
            if self.flush_refusal:
                text += f": {self.flush_refusal}"
        return text


class _Pass:
    """The bookkeeping of one pass, shared by the blocking and the
    asynchronous drivers: which step is next under the budget, what each
    one cost, the lines in the log."""

    def __init__(self, plan: WarmupPlan, budget_secs: float, log: logging.Logger) -> None:
        self.plan = plan
        self.budget_secs = budget_secs
        self.log = log
        self.report = WarmupReport(plan)
        self._started = time.monotonic()
        self._failed_in_a_row = 0
        # The step that opened the current run of whole failures: the one the
        # end line names. A partly failed step neither extends nor ends the
        # run, so the run's length does not locate its first step.
        self._first_failed: int | None = None
        self._remainders = sum(1 for step in plan.steps if step.kind == "remainder")
        self._remainders_done = 0
        self._remainder_secs = 0.0
        self._remainder_failures = 0
        log.info(
            "Planned warm-up starts (%s; budget %.0f s%s): readiness is held until it ran",
            plan.describe(),
            budget_secs,
            f"; {plan.inputs.source}" if plan.inputs.source else "",
        )

    def next_step(self) -> WarmupStep | None:
        done = len(self.report.results)
        if done >= len(self.plan.steps):
            return None
        if time.monotonic() - self._started > self.budget_secs:
            self.report.budget_hit = True
            return None
        if self._failed_in_a_row >= CONSECUTIVE_FAILED_STEPS:
            # The end line names the first of them: the shape the engine
            # refused, or the step at which it stopped answering.
            opener = (
                done - self._failed_in_a_row if self._first_failed is None else self._first_failed
            )
            first = self.plan.steps[opener].name
            self.report.stopped = f"{CONSECUTIVE_FAILED_STEPS} steps failed in a row, from {first}"
            return None
        return self.plan.steps[done]

    def remaining(self) -> float:
        """What is left of the budget, :data:`REQUEST_TIMEOUT_FLOOR_SECS` at least."""
        return max(
            self.budget_secs - (time.monotonic() - self._started), REQUEST_TIMEOUT_FLOOR_SECS
        )

    def timeout(self, request_timeout_secs: float) -> float:
        """A request's timeout: the caller's, clamped to what is left of the
        budget, so a step that hangs near the budget's end cannot hold
        readiness past it."""
        return min(request_timeout_secs, self.remaining())

    def record(self, step: WarmupStep, secs: float, errors: list[BaseException]) -> None:
        self.report.results.append(StepResult(step, secs, len(errors)))
        index = len(self.report.results)
        remainder = step.kind == "remainder"
        if errors:
            if len(errors) == len(step.requests):
                if self._failed_in_a_row == 0:
                    self._first_failed = index - 1
                self._failed_in_a_row += 1
            self.log.warning(
                "Warm-up step %d/%d %s: %d of %d requests failed in %.0f ms (first: %s)",
                index,
                len(self.plan.steps),
                step.name,
                len(errors),
                len(step.requests),
                secs * 1000,
                errors[0],
            )
        else:
            self._failed_in_a_row = 0
            self._first_failed = None
            # A chunk remainder is one of thousands of one-token requests: its
            # own line is DEBUG, and the remainders' progress is one INFO line
            # every so many (:data:`REMAINDER_PROGRESS_EVERY`) and at their end.
            self.log.log(
                logging.DEBUG if remainder else logging.INFO,
                "Warm-up step %d/%d %s: %.0f ms",
                index,
                len(self.plan.steps),
                step.name,
                secs * 1000,
            )
        if remainder:
            self._remainder_done(secs, bool(errors))

    def _remainder_done(self, secs: float, failed: bool) -> None:
        self._remainders_done += 1
        self._remainder_secs += secs
        self._remainder_failures += 1 if failed else 0
        done = self._remainders_done
        if done % REMAINDER_PROGRESS_EVERY == 0 or done == self._remainders:
            self._remainder_progress()

    def _remainder_progress(self) -> None:
        done = self._remainders_done
        self.log.info(
            "Warm-up chunk remainders: %d of %d done in %.0f s (%.0f ms each, %d failed)%s",
            done,
            self._remainders,
            self._remainder_secs,
            self._remainder_secs * 1000 / done,
            self._remainder_failures,
            " (every count up to the step used once)" if done == self._remainders else "",
        )

    def finish(self, flushed: bool | None, flush_refusal: str | None = None) -> WarmupReport:
        self.report.secs = time.monotonic() - self._started
        self.report.flushed = flushed
        self.report.flush_refusal = flush_refusal
        # The remainders the budget or a stop cut short: their last partial
        # batch reported once, as the periodic line would have.
        done = self._remainders_done
        if done and done != self._remainders and done % REMAINDER_PROGRESS_EVERY:
            self._remainder_progress()
        self.log.info("Planned warm-up over: %s", self.report.summary())
        return self.report


def _refusal(error: BaseException) -> str:
    """The words of a refused flush, for the log and the end line: a gRPC
    status' details when the error carries them (a status' ``str`` runs over
    several lines), else its code's name, else the error itself. Whatever the
    source, the words are one line: runs of whitespace, newlines included,
    read as one space (an engine's multi-line message would otherwise split
    the end line)."""
    details = getattr(error, "details", None)
    if callable(details):
        with contextlib.suppress(Exception):
            if text := _one_line(str(details() or "")):
                return text
    # A status with empty details: its code's name keeps the reason to one
    # line (the status' ``str`` is the multi-line repr).
    code = getattr(error, "code", None)
    if callable(code):
        with contextlib.suppress(Exception):
            if (status := code()) is not None:
                return _one_line(str(getattr(status, "name", status)))
    return _one_line(str(error)) or type(error).__name__


def _one_line(text: str) -> str:
    """``text`` with every run of whitespace, newlines included, as one space."""
    return " ".join(text.split())


def _flush(wire: WarmupWire, log: logging.Logger) -> tuple[bool, str | None]:
    """The engine's cache flushed, best effort; the outcome is logged, never
    raised. Returns whether it was, and the refusal's words when it was not."""
    try:
        return bool(wire.flush_cache()), None
    except Exception as error:  # noqa: BLE001 -- the flush is a courtesy to the next requests
        refusal = _refusal(error)
        log.info("The warm-up's cache flush was not accepted: %s", refusal)
        return False, refusal


def run_warmup(
    plan: WarmupPlan,
    wire: WarmupWire,
    *,
    budget_secs: float = DEFAULT_BUDGET_SECS,
    request_timeout_secs: float = REQUEST_TIMEOUT_SECS,
    log: logging.Logger = logger,
) -> WarmupReport:
    """Run ``plan`` over a blocking ``wire``: the requests of a step together
    (one thread each), the steps in order, under ``budget_secs`` (a request's
    timeout is ``request_timeout_secs`` clamped to what is left of the
    budget); the engine's cache flushed at the end. Returns the report."""
    run = _Pass(plan, budget_secs, log)
    while (step := run.next_step()) is not None:
        started = time.monotonic()
        timeout = run.timeout(request_timeout_secs)
        errors: list[BaseException] = []
        if len(step.requests) == 1:
            try:
                wire.generate(step.requests[0], timeout)
            except Exception as error:  # noqa: BLE001 -- counted, the pass goes on
                errors.append(error)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(step.requests)) as pool:
                futures = []
                for request in step.requests:
                    try:
                        futures.append(pool.submit(wire.generate, request, timeout))
                    except RuntimeError as error:
                        # The host refused another thread (a pid limit under a
                        # wide window): that request failed; the step's others
                        # and the steps after it go on.
                        errors.append(error)
                for future in futures:
                    error = future.exception()
                    if error is not None:
                        errors.append(error)
        run.record(step, time.monotonic() - started, errors)
    return run.finish(*(_flush(wire, log) if run.report.results else (None, None)))


async def run_warmup_async(
    plan: WarmupPlan,
    wire: WarmupWire,
    *,
    budget_secs: float = DEFAULT_BUDGET_SECS,
    request_timeout_secs: float = REQUEST_TIMEOUT_SECS,
    log: logging.Logger = logger,
) -> WarmupReport:
    """:func:`run_warmup` over a wire whose methods are coroutines (an engine
    driven in this process, on its event loop)."""
    run = _Pass(plan, budget_secs, log)
    while (step := run.next_step()) is not None:
        started = time.monotonic()
        timeout = run.timeout(request_timeout_secs)
        outcomes = await asyncio.gather(
            *(wire.generate(request, timeout) for request in step.requests),
            return_exceptions=True,
        )
        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        run.record(step, time.monotonic() - started, errors)
    flushed = refusal = None
    if run.report.results:
        try:
            flushed = bool(await wire.flush_cache())
        except Exception as error:  # noqa: BLE001 -- the flush is a courtesy to the next requests
            flushed, refusal = False, _refusal(error)
            log.info("The warm-up's cache flush was not accepted: %s", refusal)
    return run.finish(flushed, refusal)


def plan_for(
    setting: WarmupSetting,
    inputs: WarmupInputs | None,
    *,
    knob: str,
    log: logging.Logger = logger,
) -> WarmupPlan | None:
    """The plan a launcher runs for ``setting`` and ``inputs``, or ``None``
    with the reason logged: the knob is off, the engine or model takes no
    warm-up (no inputs), or the configuration leaves nothing to plan."""
    if not setting.enabled:
        log.info("Planned warm-up off (%s=off)", knob)
        return None
    if inputs is None:
        log.info("Planned warm-up: none for this engine or model")
        return None
    plan = plan_warmup(inputs, setting.level)
    if inputs.every_prefill_length and setting.level != LEVEL_FULL:
        log.warning(
            "Planned warm-up at the %s level plans the length classes alone and leaves out the "
            "chunk remainders this engine asks for: a prompt length's first use during serving "
            "stays a hazard until a start at the full level",
            setting.level,
        )
    if not plan.steps:
        log.info(
            "Planned warm-up: nothing to plan (window %d, step %d, context %d, vocabulary %d)",
            inputs.max_num_seqs,
            inputs.max_num_batched_tokens,
            inputs.max_model_len,
            inputs.vocab_size,
        )
        return None
    return plan


WarmupRunner = Callable[[], WarmupReport | Awaitable[WarmupReport]]


def describe_steps(steps: Sequence[WarmupStep]) -> list[str]:
    """The step names, for tests and tools."""
    return [step.name for step in steps]
