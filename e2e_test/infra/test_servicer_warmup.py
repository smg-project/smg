"""The servicer warm-up reader: the log lines it parses, the watch it keeps
over a spawned worker, the knobs it names, and the worker's one-shot health
probe the watch relies on."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
from infra import ConnectionMode
from infra import worker as worker_module
from infra.servicer_warmup import (
    LEVEL_FAST,
    LEVEL_FULL,
    SERVICER_LOG_ENGINE_ARGS,
    SERVICER_WARMUP_ENV,
    WarmupEnd,
    WarmupStart,
    warmup_end,
    warmup_skipped,
    warmup_start,
    watch_warmup,
)
from infra.worker import Worker

_SERVICER = Path(__file__).resolve().parents[2] / "grpc_servicer" / "smg_grpc_servicer"

START_LINE = (
    "(MainProcess pid=9053) INFO 10-22 19:20:59 [warmup.py:440] Planned warm-up starts "
    "(level full: 215 steps (1 decode staircase, 192 prefill shapes, 22 measured again), "
    "1152519 prompt tokens and 526038 output tokens over a window of 1024 and a step of "
    "16384; budget 300 s; window, step and capture sizes from the engine's resolved "
    "configuration): readiness is held until it ran\n"
)
END_LINE = (
    "(MainProcess pid=9053) INFO 10-22 19:21:24 [warmup.py:495] Planned warm-up over: "
    "215 of 215 steps covered in 25.7 s (0 failed requests)\n"
)

# The same two lines as the other engines' processes write them, with the lines
# around them: the engine's root logger format under the Python servicer, and
# the Rust servicer's own (coloured) lines between them under the Rust one.
SGLANG_PYTHON_LOG = (
    "[2026-10-23 10:09:39] Servicer implementation: python (source=default)\n"
    "[2026-10-23 10:10:19] Planned warm-up starts (level full: 175 steps (1 decode staircase, "
    "152 prefill shapes, 22 measured again), 1136255 prompt tokens and 8558 output tokens over "
    "a window of 0 and a step of 16384; budget 300 s; window and step from the server arguments "
    "(the engine's default step when unset)): readiness is held until it ran\n"
    "[2026-10-23 10:10:20] Warm-up step 1/175 decode batch 128 down to 1 (128 requests): 718 ms\n"
    "[2026-10-23 10:10:33] Warm-up step 175/175 prefill 16384 tokens again: 200 ms\n"
    "[2026-10-23 10:10:33] Planned warm-up over: 175 of 175 steps covered in 14.6 s "
    "(0 failed requests)\n"
    "[2026-10-23 10:10:33] The server is fired up and ready to roll!\n"
)
SGLANG_RUST_LOG = (
    "[2026-10-23 10:10:48] Servicer implementation: rust (source=env)\n"
    "[2026-10-23 10:10:48] Rust SGLang servicer listening on 127.0.0.1:41455 (scheduler handshake "
    "tcp://127.0.0.1:42879, 1 DP rank(s))\n"
    "\x1b[2m2026-10-23T10:11:26.307699Z\x1b[0m \x1b[32m INFO\x1b[0m "
    "\x1b[2mengine_servicer::sglang::engine\x1b[0m\x1b[2m:\x1b[0m SGLang scheduler connected "
    "\x1b[3mhandshake\x1b[0m\x1b[2m=\x1b[0mtcp://127.0.0.1:42879 \x1b[3mengines\x1b[0m\x1b[2m=\x1b[0m1\n"
    "[2026-10-23 10:11:26] Planned warm-up starts (level full: 175 steps (1 decode staircase, "
    "152 prefill shapes, 22 measured again), 1136255 prompt tokens and 8558 output tokens over "
    "a window of 0 and a step of 16384; budget 300 s; window and step from the server arguments "
    "(the engine's default step when unset)): readiness is held until it ran\n"
    "[2026-10-23 10:11:27] Warm-up step 1/175 decode batch 128 down to 1 (128 requests): 811 ms\n"
    "[2026-10-23 10:11:41] Planned warm-up over: 175 of 175 steps covered in 14.8 s "
    "(0 failed requests)\n"
)
TOKENSPEED_PYTHON_LOG = (
    "2026-10-23 10:54:53,259 [smg_grpc_servicer.tokenspeed] INFO Servicer implementation: "
    "python (source=default)\n"
    "[2026-10-23 10:58:52] Planned warm-up starts (level full: 109 steps (1 decode staircase, "
    "87 prefill shapes, 21 measured again), 321406 prompt tokens and 13148 output tokens over "
    "a window of 160 and a step of 8192; budget 300 s; window and step from the server "
    "arguments (the engine's default step when unset)): readiness is held until it ran\n"
    "[2026-10-23 10:58:53] Warm-up step 1/109 decode batch 160 down to 1 (160 requests): 916 ms\n"
    "[2026-10-23 10:58:57] Planned warm-up over: 109 of 109 steps covered in 5.6 s "
    "(0 failed requests)\n"
    "[2026-10-23 10:58:57] TokenSpeed gRPC server is ready to serve\n"
)
TOKENSPEED_RUST_LOG = (
    "2026-10-23 10:59:15,053 [smg_grpc_servicer.rust_lifecycle] INFO Rust TokenSpeed servicer "
    "listening on 127.0.0.1:38167 (scheduler handshake tcp://127.0.0.1:52719, 1 DP rank(s))\n"
    "\x1b[2m2026-10-23T11:00:18.672282Z\x1b[0m \x1b[32m INFO\x1b[0m "
    "\x1b[2mengine_servicer::tokenspeed::engine\x1b[0m\x1b[2m:\x1b[0m TokenSpeed scheduler "
    "connected \x1b[3mhandshake\x1b[0m\x1b[2m=\x1b[0mtcp://127.0.0.1:52719\n"
    "[2026-10-23 11:00:19] Planned warm-up starts (level full: 109 steps (1 decode staircase, "
    "87 prefill shapes, 21 measured again), 321406 prompt tokens and 13148 output tokens over "
    "a window of 160 and a step of 8192; budget 300 s; window and step from the server "
    "arguments (the engine's default step when unset)): readiness is held until it ran\n"
    "[2026-10-23 11:00:25] Planned warm-up over: 109 of 109 steps covered in 6.3 s "
    "(0 failed requests); the engine's cache was not flushed (the warm-up's blocks stay cached)\n"
)


# ---------------------------------------------------------------------------
# The lines
# ---------------------------------------------------------------------------


def test_the_start_line_gives_the_level_the_step_count_and_the_budget():
    assert warmup_start("engine lines\n" + START_LINE) == WarmupStart(LEVEL_FULL, 215, 300.0)
    assert warmup_start("nothing yet") is None


@pytest.mark.parametrize(
    ("log", "steps", "secs"),
    [
        pytest.param(SGLANG_PYTHON_LOG, 175, 14.6, id="sglang-python-servicer"),
        pytest.param(SGLANG_RUST_LOG, 175, 14.8, id="sglang-rust-servicer-coloured-lines-between"),
        pytest.param(TOKENSPEED_PYTHON_LOG, 109, 5.6, id="tokenspeed-python-servicer"),
        pytest.param(TOKENSPEED_RUST_LOG, 109, 6.3, id="tokenspeed-rust-servicer-unflushed-end"),
    ],
)
def test_the_lines_are_read_in_the_other_engines_log_formats(log, steps, secs):
    assert warmup_start(log) == WarmupStart(LEVEL_FULL, steps, 300.0)
    end = warmup_end(log)
    assert end == WarmupEnd(steps, steps, secs, 0, budget_hit=False, stopped=False)
    assert end.complete
    assert warmup_skipped(log) is None


def test_a_fast_start_line_with_a_set_budget_and_data_parallel_ranks_parses_too():
    line = (
        "Planned warm-up starts (level fast: 119 steps (2 decode staircases, 96 prefill "
        "shapes, 22 measured again), 149511 prompt tokens and 525942 output tokens over a "
        "window of 1024 and a step of 16384 on each of 2 data-parallel ranks; budget 45 s): "
        "readiness is held until it ran"
    )
    assert warmup_start(line) == WarmupStart(LEVEL_FAST, 119, 45.0)


def test_the_end_line_gives_the_coverage_and_a_complete_pass():
    end = warmup_end(START_LINE + "steps\n" + END_LINE)
    assert end == WarmupEnd(215, 215, 25.7, 0, budget_hit=False, stopped=False)
    assert end.complete
    assert warmup_end(START_LINE) is None


@pytest.mark.parametrize(
    ("rest", "covered", "failed", "budget_hit", "stopped", "left"),
    [
        (
            "; the budget ended the pass before prefill 16384 tokens again",
            202,
            0,
            True,
            False,
            "prefill 16384 tokens again",
        ),
        ("; the budget ended the pass", 243, 0, True, False, ""),
        (
            "; the budget ended the pass before prefill 37 tokens (chunk remainder)",
            230,
            0,
            True,
            False,
            "prefill 37 tokens (chunk remainder)",
        ),
        ("; stopped: 3 steps failed in a row", 40, 3, False, True, ""),
        (
            "; the engine's cache was not flushed (the warm-up's blocks stay cached)",
            243,
            0,
            False,
            False,
            "",
        ),
    ],
)
def test_an_early_or_unflushed_end_is_told_apart(rest, covered, failed, budget_hit, stopped, left):
    line = f"Planned warm-up over: {covered} of 243 steps covered in 300.4 s ({failed} failed requests){rest}"
    end = warmup_end(line)
    assert end == WarmupEnd(
        covered, 243, 300.4, failed, budget_hit=budget_hit, stopped=stopped, left=left
    )
    assert end.complete is (covered == 243 and not budget_hit and not stopped)
    # Only the chunk remainders left: the plan's steps before them all ran.
    assert end.budget_ended_in_the_remainders is (budget_hit and left.endswith("(chunk remainder)"))


@pytest.mark.parametrize(
    "line",
    [
        "Planned warm-up off (SMG_VLLM_SERVICER_WARMUP=off)",
        "Planned warm-up: none for this engine or model",
        "Planned warm-up: nothing to plan (window 0, step 0, context 0, vocabulary 0)",
    ],
)
def test_a_servicer_that_ran_no_pass_says_so(line):
    assert warmup_skipped("engine lines\n" + line + "\nmore") == line
    assert warmup_skipped(START_LINE + END_LINE) is None


# ---------------------------------------------------------------------------
# The watch
# ---------------------------------------------------------------------------


class _FakeWorker:
    """A spawned worker as the watch sees it: a log that grows, a health answer
    that flips, a life that may end; every read advances a fake clock."""

    model_id = "m"

    def __init__(self, timeline):
        # timeline: list of (log_text, serving, alive) per poll, the last one repeats.
        self.timeline = list(timeline)
        self.polls = 0
        self.clock = 0.0
        self.slept: list[float] = []

    def _now(self):
        return self.timeline[min(self.polls, len(self.timeline) - 1)]

    def read_log(self):
        return self._now()[0]

    def grpc_serving(self):
        serving = self._now()[1]
        self.polls += 1
        return serving

    def is_alive(self):
        return self._now()[2]

    def sleep(self, secs):
        self.slept.append(secs)
        self.clock += secs


def _watch(worker, **kwargs):
    kwargs.setdefault("engine_start_secs", 100.0)
    kwargs.setdefault("poll_secs", 2.0)
    return watch_warmup(worker, clock=lambda: worker.clock, sleep=worker.sleep, **kwargs)


def test_the_watch_follows_the_pass_to_serving_and_reports_it():
    worker = _FakeWorker(
        [
            ("engine loading", False, True),
            (START_LINE, False, True),
            (START_LINE + "steps", False, True),
            (START_LINE + END_LINE, True, True),
        ]
    )
    outcome = _watch(worker)

    assert outcome.start == WarmupStart(LEVEL_FULL, 215, 300.0)
    assert outcome.end.complete and outcome.end.secs == 25.7
    assert outcome.serving_after_secs == 6.0  # three polls of 2 s before the SERVING read
    assert outcome.end_lag_secs == 0.0
    assert "level full: 215 of 215 steps in 25.7 s under a 300 s budget" in outcome.describe()
    assert "ran to its end; SERVING 6.0 s after the spawn" in outcome.describe()


def test_the_engine_start_bound_ends_a_wait_that_never_sees_a_pass():
    worker = _FakeWorker([("engine loading", False, True)])
    with pytest.raises(TimeoutError, match=r"engine's start \(10 s, no planned warm-up"):
        _watch(worker, engine_start_secs=10.0)
    assert worker.clock == 10.0


def test_the_start_line_extends_the_wait_by_the_budget_and_the_grace():
    worker = _FakeWorker([("engine loading", False, True), (START_LINE, False, True)])
    with pytest.raises(TimeoutError, match=r"planned warm-up \(budget 300 s, plus 30 s\)"):
        _watch(worker, engine_start_secs=10.0)
    # The line was read at 2 s; the budget's 300 s and the grace's 30 s count from there.
    assert worker.clock == 332.0


def test_serving_before_the_end_line_is_the_defect_the_watch_catches():
    worker = _FakeWorker([(START_LINE + "step 100/215", True, True)])
    with pytest.raises(RuntimeError, match="answered SERVING with no planned warm-up end line"):
        _watch(worker, end_grace_secs=5.0)
    assert worker.clock == 6.0  # the end grace: one-second re-reads until it is spent


def test_an_end_line_that_trails_the_flip_by_a_moment_is_accepted_with_its_lag():
    worker = _FakeWorker(
        [
            (START_LINE, True, True),  # the flip is read first
            (START_LINE, True, True),  # the re-read after 1 s: not yet
            (START_LINE + END_LINE, True, True),  # the re-read after 2 s
        ]
    )
    # The re-reads call read_log without grpc_serving; advance the timeline by hand.
    reads = iter(worker.timeline)
    worker.read_log = lambda: next(reads)[0]
    outcome = _watch(worker)
    assert outcome.end_lag_secs == 2.0
    assert outcome.end.complete


def test_a_worker_that_dies_fails_the_watch_with_its_log_tail():
    worker = _FakeWorker([(START_LINE + "CUDA error: out of memory", False, False)])
    with pytest.raises(RuntimeError, match=r"(?s)died before SERVING.*out of memory"):
        _watch(worker)


def test_a_servicer_that_ran_no_pass_fails_the_watch():
    worker = _FakeWorker([("Planned warm-up off (SMG_VLLM_SERVICER_WARMUP=off)", False, True)])
    with pytest.raises(RuntimeError, match=r"ran no pass: Planned warm-up off"):
        _watch(worker)


# ---------------------------------------------------------------------------
# The knobs
# ---------------------------------------------------------------------------


def _servicer_warmup_env(engine: str) -> str:
    """``WARMUP_ENV`` as the engine's servicer module assigns it, read off the source."""
    for node in ast.parse((_SERVICER / engine / "warmup.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "WARMUP_ENV" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"WARMUP_ENV is not assigned at the top of {engine}/warmup.py")


@pytest.mark.parametrize("engine", sorted(SERVICER_WARMUP_ENV))
def test_the_knob_names_are_the_servicers_own(engine):
    if not (_SERVICER / engine / "warmup.py").is_file():
        pytest.skip(f"{_SERVICER / engine / 'warmup.py'} is not in this tree")
    assert SERVICER_WARMUP_ENV[engine] == _servicer_warmup_env(engine)


def test_a_workers_own_knob_wins_over_the_jobs_level(monkeypatch):
    monkeypatch.setenv("SMG_VLLM_SERVICER_WARMUP", LEVEL_FAST)
    inherited = Worker(model_id="m", engine="vllm", port=50111, gpu_ids=[0])._build_env()
    assert inherited["SMG_VLLM_SERVICER_WARMUP"] == LEVEL_FAST
    own = Worker(
        model_id="m",
        engine="vllm",
        port=50111,
        gpu_ids=[0],
        extra_env={"SMG_VLLM_SERVICER_WARMUP": LEVEL_FULL},
    )._build_env()
    assert own["SMG_VLLM_SERVICER_WARMUP"] == LEVEL_FULL


# ---------------------------------------------------------------------------
# The engine's log level
# ---------------------------------------------------------------------------

# The case's model: any worker of the suite's smallest model, as the case starts it.
CASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def _log_level_flag(cmd: list[str]) -> str | None:
    """The engine's ``--log-level`` as its parser keeps it: the last one given."""
    level = None
    for flag, value in zip(cmd, cmd[1:]):
        if flag == "--log-level":
            level = value
    return level


@pytest.mark.parametrize("engine", sorted(SERVICER_WARMUP_ENV))
def test_the_followed_workers_engine_logs_at_the_level_the_pass_lines_carry(engine):
    """The legs' workers run the engines that take a ``--log-level`` at
    ``warning``, where the servicer's INFO lines are not written; the worker a
    case follows carries the table's arguments and runs at ``info``. An engine
    the harness leaves at its own default has nothing to add."""
    as_the_legs = Worker(
        model_id=CASE_MODEL, engine=engine, port=50111, gpu_ids=[0], mode=ConnectionMode.GRPC
    )
    followed = dataclasses.replace(as_the_legs, extra_engine_args=SERVICER_LOG_ENGINE_ARGS[engine])
    legs_level = _log_level_flag(as_the_legs._build_cmd())
    if legs_level is None:
        assert SERVICER_LOG_ENGINE_ARGS[engine] == []
    else:
        assert legs_level == "warning"
        assert _log_level_flag(followed._build_cmd()) == "info"


# ---------------------------------------------------------------------------
# The worker's one-shot probe under the health wait
# ---------------------------------------------------------------------------


def test_the_health_wait_polls_the_probe_until_serving(monkeypatch):
    answers = iter([False, False, True])
    clock = [0.0]
    monkeypatch.setattr(worker_module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        worker_module.time, "sleep", lambda secs: clock.__setitem__(0, clock[0] + secs)
    )
    monkeypatch.setattr(Worker, "grpc_serving", lambda self: next(answers))
    monkeypatch.setattr(Worker, "is_alive", lambda self: True)
    Worker(model_id="m", engine="vllm", port=50111, gpu_ids=[0])._wait_grpc_healthy(timeout=10)
    assert clock[0] == 2 * worker_module.HEALTH_CHECK_INTERVAL


def test_the_health_wait_ends_on_its_bound_and_on_a_dead_worker(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(worker_module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        worker_module.time, "sleep", lambda secs: clock.__setitem__(0, clock[0] + secs)
    )
    monkeypatch.setattr(Worker, "grpc_serving", lambda self: False)
    monkeypatch.setattr(Worker, "is_alive", lambda self: True)
    worker = Worker(model_id="m", engine="vllm", port=50111, gpu_ids=[0])
    with pytest.raises(TimeoutError, match="did not become healthy within 5s"):
        worker._wait_grpc_healthy(timeout=5)

    monkeypatch.setattr(Worker, "is_alive", lambda self: False)
    with pytest.raises(RuntimeError, match="died during startup"):
        worker._wait_grpc_healthy(timeout=5)


def test_a_pass_that_ran_every_step_with_a_failed_request_is_described_as_such():
    """A step with a failed request is not covered, so the pass is not complete
    though nothing ended it early: the description says what happened instead
    of an early end without a reason."""
    from infra.servicer_warmup import WarmupEnd, WarmupOutcome, WarmupStart

    start = WarmupStart("full", 10, 300.0)
    failed = WarmupOutcome(
        start,
        WarmupEnd(
            covered=8, planned=10, secs=12.0, failed_requests=3, budget_hit=False, stopped=False
        ),
        6.0,
        0.0,
    )
    assert not failed.end.complete
    assert "ran every step, 2 with failed requests" in failed.describe()
    assert "ended early" not in failed.describe()
    budget = WarmupOutcome(
        start,
        WarmupEnd(
            covered=8, planned=10, secs=12.0, failed_requests=0, budget_hit=True, stopped=False
        ),
        6.0,
        0.0,
    )
    assert "ended early on the budget" in budget.describe()


def test_a_pass_the_budget_ended_in_the_chunk_remainders_is_described_as_such():
    """An engine that plans every prefill length runs the chunk remainders last,
    under the same budget: a pass the budget ends there covered every step before
    them, and the description names the remainder it stopped before."""
    from infra.servicer_warmup import WarmupEnd, WarmupOutcome, WarmupStart

    start = WarmupStart("full", 8355, 900.0)
    cut = WarmupOutcome(
        start,
        WarmupEnd(
            covered=6210,
            planned=8355,
            secs=900.2,
            failed_requests=0,
            budget_hit=True,
            stopped=False,
            left="prefill 5996 tokens (chunk remainder)",
        ),
        912.0,
        0.0,
    )
    assert not cut.end.complete
    assert cut.end.budget_ended_in_the_remainders
    assert "ended on the budget in the chunk remainders (before prefill 5996 tokens" in (
        cut.describe()
    )
    before = WarmupEnd(
        covered=200,
        planned=8355,
        secs=900.2,
        failed_requests=0,
        budget_hit=True,
        stopped=False,
        left="prefill 16384 tokens again",
    )
    assert not before.budget_ended_in_the_remainders
