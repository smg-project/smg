"""The planned warm-up (``smg_grpc_servicer.warmup``): the knob, the plan's
shapes and bounds, the pass under its budget, the engines' wires against an
in-process stand-in for each engine's gRPC service, the lifecycle's hold on
readiness until the pass ran, and the Python servicer's health hold."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import logging
import re
import sys
import threading
import time
import types
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
from smg_grpc_proto import sglang_scheduler_pb2 as sgl_pb
from smg_grpc_proto import sglang_scheduler_pb2_grpc as sgl_grpc
from smg_grpc_proto import tokenspeed_scheduler_pb2 as ts_pb
from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc as ts_grpc
from smg_grpc_proto import vllm_engine_pb2 as vllm_pb
from smg_grpc_proto import vllm_engine_pb2_grpc as vllm_grpc
from smg_grpc_proto.generated import common_pb2
from smg_grpc_servicer import rust_lifecycle, warmup
from smg_grpc_servicer.sglang import warmup as sglang_warmup
from smg_grpc_servicer.tokenspeed import warmup as tokenspeed_warmup
from smg_grpc_servicer.vllm import warmup as vllm_warmup
from smg_grpc_servicer.warmup import (
    DEFAULT_BUDGET_SECS,
    WarmupInputs,
    hybrid_buckets,
    odd_variants,
    parse_setting,
    plan_warmup,
    prefill_classes,
    run_warmup,
    token_ids,
)

# The issue's engine: a running window of 128, an admission step of 16,384
# tokens, a context of 128k, a 129k vocabulary, the derived capture list.
CAPTURE_SIZES = tuple([1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16)))
ISSUE = WarmupInputs(
    max_num_seqs=128,
    max_num_batched_tokens=16384,
    max_model_len=131072,
    vocab_size=129280,
    capture_sizes=CAPTURE_SIZES,
)
SMALL = WarmupInputs(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256, vocab_size=32000)


# ---------------------------------------------------------------------------
# The knob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, enabled, level, budget",
    [
        (None, True, "full", DEFAULT_BUDGET_SECS),
        ("", True, "full", DEFAULT_BUDGET_SECS),
        ("  FULL ", True, "full", DEFAULT_BUDGET_SECS),
        ("fast", True, "fast", DEFAULT_BUDGET_SECS),
        ("off", False, "full", DEFAULT_BUDGET_SECS),
        ("0", False, "full", DEFAULT_BUDGET_SECS),
        ("false", False, "full", DEFAULT_BUDGET_SECS),
        ("120", True, "full", 120.0),
        ("fast:30", True, "fast", 30.0),
        ("full:0.5", True, "full", 0.5),
    ],
)
def test_the_knob_reads_a_level_a_budget_or_both(text, enabled, level, budget):
    setting = parse_setting(text, name="SMG_X_SERVICER_WARMUP")
    assert (setting.enabled, setting.level, setting.budget_secs) == (enabled, level, budget)


@pytest.mark.parametrize(
    "text", ["warm", "fast:", "fast:abc", "full:-1", "medium:10", "inf", "nan"]
)
def test_a_value_the_knob_does_not_take_names_the_variable(text):
    with pytest.raises(ValueError) as caught:
        parse_setting(text, name="SMG_X_SERVICER_WARMUP")
    assert "SMG_X_SERVICER_WARMUP" in str(caught.value) and repr(text) in str(caught.value)


def test_an_engine_module_may_state_its_own_default_budget():
    knob = "SMG_X_SERVICER_WARMUP"
    assert parse_setting(None, name=knob, default_budget_secs=900.0).budget_secs == 900.0
    assert parse_setting("fast", name=knob, default_budget_secs=900.0).budget_secs == 900.0
    assert parse_setting("120", name=knob, default_budget_secs=900.0).budget_secs == 120.0
    assert parse_setting(None, name=knob).budget_secs == DEFAULT_BUDGET_SECS == 300.0
    with pytest.raises(ValueError) as caught:
        parse_setting("warm", name=knob, default_budget_secs=900.0)
    assert "full:900" in str(caught.value)
    read = rust_lifecycle.read_warmup_setting(knob, environ={}, default_budget_secs=900.0)
    assert read.budget_secs == 900.0
    # TokenSpeed's plan holds the chunk remainders: its module states 900 s;
    # the other engines keep the shared default.
    assert tokenspeed_warmup.DEFAULT_BUDGET_SECS == 900.0
    assert rust_lifecycle.RustLauncher.warmup_default_budget_secs == DEFAULT_BUDGET_SECS


def test_the_tokenspeed_launcher_states_its_own_default_budget():
    # The launcher module needs the engine extra's wire packages.
    tokenspeed_rust = pytest.importorskip("smg_grpc_servicer.tokenspeed.rust")
    assert tokenspeed_rust.TokenSpeedLauncher.warmup_default_budget_secs == 900.0


def test_read_warmup_setting_is_a_configuration_error_before_the_port():
    with pytest.raises(rust_lifecycle.ServicerConfigError) as caught:
        rust_lifecycle.read_warmup_setting(
            "SMG_X_SERVICER_WARMUP", environ={"SMG_X_SERVICER_WARMUP": "warm"}
        )
    assert "SMG_X_SERVICER_WARMUP" in str(caught.value)
    assert rust_lifecycle.read_warmup_setting("", environ={}).enabled is False
    assert rust_lifecycle.read_warmup_setting("SMG_X_SERVICER_WARMUP", environ={}).enabled is True


def test_the_servicers_own_listener_is_dialed_on_loopback():
    assert rust_lifecycle.servicer_loopback("[::]:50051") == "[::1]:50051"
    assert rust_lifecycle.servicer_loopback("0.0.0.0:50051") == "127.0.0.1:50051"
    assert rust_lifecycle.servicer_loopback("10.1.2.3:7") == "10.1.2.3:7"


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_the_bucket_grid_is_the_tuners_hybrid_grid_plus_the_limit():
    assert hybrid_buckets(16384) == [
        *[1, 2, 4, 8, 16, 32, 64, 128, 256],
        *[512, 768, 1024, 1280, 1536, 1792, 2048],
        *[2560, 3072, 3584, 4096],
        8192,
        16384,
    ]
    assert hybrid_buckets(3000) == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        768,
        1024,
        1280,
        1536,
        1792,
        2048,
        2560,
        3000,
    ]
    assert hybrid_buckets(1) == [1]
    assert hybrid_buckets(0) == []


def test_the_prefill_classes_are_every_128_token_class_up_to_the_step():
    assert prefill_classes(16384) == list(range(128, 16385, 128))
    assert prefill_classes(300) == [128, 256, 300]
    assert prefill_classes(100) == [100]


def test_the_odd_variants_sit_one_above_the_aligned_buckets():
    assert odd_variants([1, 8, 16, 32, 256], 256) == [17, 33]
    assert odd_variants([16, 32], 257) == [17, 33]


def test_token_ids_are_unique_per_request_and_from_the_ordinary_region():
    first, second = token_ids(64, 0, 129280), token_ids(64, 1, 129280)
    assert first != second and first[:16] != second[:16]
    assert token_ids(64, 0, 129280) == first  # deterministic per ordinal
    assert all(1000 <= value < 31000 for value in first + second)
    # A small vocabulary draws from the whole of it.
    assert all(0 <= value < 64 for value in token_ids(32, 3, 64))


def test_the_full_plan_covers_the_staircase_every_class_the_capture_sizes_and_the_odd_variants():
    plan = plan_warmup(ISSUE, "full")
    names = [step.name for step in plan.steps]
    staircase = plan.steps[0]
    assert staircase.kind == "decode" and staircase.name.startswith("decode batch 128 down to 1")
    assert [request.max_tokens for request in staircase.requests] == list(range(2, 130))
    assert {len(request.token_ids) for request in staircase.requests} == {
        warmup.STAIRCASE_PROMPT_TOKENS
    }
    prefill = [step for step in plan.steps if step.kind == "prefill"]
    lengths = [len(step.requests[0].token_ids) for step in prefill]
    assert lengths == sorted(lengths)
    assert set(range(128, 16385, 128)) <= set(lengths)  # every class
    assert set(CAPTURE_SIZES) <= set(lengths)  # every capture size
    assert {17, 33, 65, 129, 257, 513, 8193} <= set(lengths)  # off the alignment
    assert all(step.requests[0].max_tokens == 1 for step in prefill)
    again = [step for step in plan.steps if step.kind == "again"]
    assert [len(step.requests[0].token_ids) for step in again] == hybrid_buckets(16384)
    assert names[-1] == "prefill 16384 tokens again"
    assert plan.steps.index(again[0]) > plan.steps.index(prefill[-1])
    assert 1_100_000 < plan.prompt_tokens < 1_200_000  # the ladder's cost, in tokens


def test_the_fast_plan_keeps_the_grid_and_drops_the_classes_between():
    fast, full = plan_warmup(ISSUE, "fast"), plan_warmup(ISSUE, "full")
    fast_lengths = {
        len(step.requests[0].token_ids) for step in fast.steps if step.kind == "prefill"
    }
    assert set(hybrid_buckets(16384)) <= fast_lengths
    full_lengths = {len(s.requests[0].token_ids) for s in full.steps if s.kind == "prefill"}
    assert (
        640 not in fast_lengths and 640 in full_lengths
    )  # a class off the grid and the capture list
    assert len(fast.steps) < len(full.steps) and fast.prompt_tokens < full.prompt_tokens / 5


def test_every_request_of_a_plan_has_its_own_tokens():
    plan = plan_warmup(ISSUE, "full")
    heads = [request.token_ids[:8] for step in plan.steps for request in step.requests]
    assert len(heads) == len(set(heads))
    ids = [request.request_id for step in plan.steps for request in step.requests]
    assert len(ids) == len(set(ids))


def test_every_request_id_has_the_form_the_servicer_admits_while_readiness_is_held():
    # The Rust servicers admit only `smg-warmup-<ordinal>` while readiness is
    # held for the warm-up (WARMUP_REQUEST_ID_PREFIX / is_warmup_request_id in
    # the engine servicer crate's request registry): a rename on either side
    # would have every warm-up request refused and the pass count only errors.
    plan = plan_warmup(ISSUE, "full")
    ids = [request.request_id for step in plan.steps for request in step.requests]
    assert ids and all(re.fullmatch(r"smg-warmup-[0-9]+", request_id) for request_id in ids)


def test_a_prompt_never_exceeds_the_step_nor_the_context_less_its_output():
    tight = WarmupInputs(
        max_num_seqs=8, max_num_batched_tokens=4096, max_model_len=1000, vocab_size=100
    )
    plan = plan_warmup(tight, "full")
    for step in plan.steps:
        for request in step.requests:
            assert len(request.token_ids) <= 4096
            assert len(request.token_ids) + request.max_tokens <= 1000
    lengths = [len(step.requests[0].token_ids) for step in plan.steps if step.kind == "prefill"]
    assert max(lengths) == 999  # the context's bound stands in for the step


def test_a_small_engine_gets_a_small_plan_in_the_same_order():
    plan = plan_warmup(SMALL, "full")
    assert [step.name for step in plan.steps] == [
        "decode batch 4 down to 1 (4 requests)",
        "prefill 1 tokens",
        "prefill 2 tokens",
        "prefill 4 tokens",
        "prefill 8 tokens",
        "prefill 16 tokens",
        "prefill 17 tokens",
        "prefill 32 tokens",
        "prefill 33 tokens",
        "prefill 64 tokens",
        "prefill 1 tokens again",
        "prefill 2 tokens again",
        "prefill 4 tokens again",
        "prefill 8 tokens again",
        "prefill 16 tokens again",
        "prefill 32 tokens again",
        "prefill 64 tokens again",
    ]


def test_an_unknown_window_takes_the_default_and_an_unknown_step_plans_nothing():
    plan = plan_warmup(WarmupInputs(0, 2048, 4096, 32000), "fast")
    assert (
        plan.steps[0].name
        == f"decode batch {warmup.DEFAULT_WINDOW} down to 1 ({warmup.DEFAULT_WINDOW} requests)"
    )
    assert plan_warmup(WarmupInputs(8, 0, 4096, 32000)).steps == ()
    assert plan_warmup(WarmupInputs(8, 2048, 4096, 0)).steps == ()
    with pytest.raises(ValueError):
        plan_warmup(SMALL, "medium")


def test_under_data_parallelism_every_rank_gets_the_staircase_and_every_shape():
    one = plan_warmup(SMALL, "fast")
    two = plan_warmup(dataclasses.replace(SMALL, data_parallel_size=2), "fast")
    staircases = [step for step in two.steps if step.kind == "decode"]
    assert [step.name for step in staircases] == [
        "decode batch 4 down to 1 (4 requests on rank 0)",
        "decode batch 4 down to 1 (4 requests on rank 1)",
    ]
    for rank, step in enumerate(staircases):
        assert {request.data_parallel_rank for request in step.requests} == {rank}
        assert [request.max_tokens for request in step.requests] == [2, 3, 4, 5]
    shapes = [step for step in two.steps if step.kind != "decode"]
    assert [step.name for step in shapes] == [s.name for s in one.steps if s.kind != "decode"]
    for step in shapes:
        assert [request.data_parallel_rank for request in step.requests] == [0, 1]
        assert len({len(r.token_ids) for r in step.requests}) == 1  # the same shape on every rank
        assert len({r.token_ids for r in step.requests}) == 2  # its own tokens on each
    assert len(two.steps) == len(one.steps) + 1 and two.prompt_tokens == 2 * one.prompt_tokens
    assert two.describe().endswith("on each of 2 data-parallel ranks")
    assert "2 decode staircases" in two.describe()
    # One rank: nothing is pinned and the plan reads as before.
    assert all(r.data_parallel_rank is None for step in one.steps for r in step.requests)
    assert "rank" not in one.describe() and "1 decode staircase," in one.describe()


def test_plan_for_says_why_nothing_is_planned(caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.warmup")
    off = warmup.WarmupSetting(False)
    assert warmup.plan_for(off, SMALL, knob="K") is None
    assert warmup.plan_for(warmup.WarmupSetting(True), None, knob="K") is None
    assert warmup.plan_for(warmup.WarmupSetting(True), WarmupInputs(8, 0, 0, 0), knob="K") is None
    assert warmup.plan_for(warmup.WarmupSetting(True, "fast"), SMALL, knob="K").level == "fast"
    messages = [record.getMessage() for record in caplog.records]
    assert any("off (K=off)" in m for m in messages)
    assert any("none for this engine or model" in m for m in messages)
    assert any("nothing to plan" in m for m in messages)


def test_the_spread_order_visits_every_count_once_and_every_prefix_is_a_uniform_spread():
    order = warmup.spread_order(64)
    assert sorted(order) == list(range(1, 65))
    assert order[:4] == [64, 32, 16, 48]
    assert set(order[:8]) == set(range(8, 65, 8))
    assert set(order[:16]) == set(range(4, 65, 4))
    assert set(order[:32]) == set(range(2, 65, 2))
    # A count off a power of two: every count once, the top power of two first.
    assert sorted(warmup.spread_order(100)) == list(range(1, 101))
    assert warmup.spread_order(100)[:3] == [64, 32, 96]
    assert warmup.spread_order(1) == [1] and warmup.spread_order(0) == []


def test_an_engine_that_asks_for_every_prefill_length_gets_the_remainders_last_in_spread_order():
    plan = plan_warmup(dataclasses.replace(SMALL, every_prefill_length=True), "full")
    ladder = [s.requests[0].prompt_tokens for s in plan.steps if s.kind == "prefill"]
    remainders = [s for s in plan.steps if s.kind == "remainder"]
    lengths = [s.requests[0].prompt_tokens for s in remainders]
    # Every count up to the step is a prefill shape of the plan, exactly once ...
    assert sorted(ladder + lengths) == list(range(1, 65))
    # ... the remainders after the measured second pass, in the spread order ...
    last_again = max(i for i, s in enumerate(plan.steps) if s.kind == "again")
    assert plan.steps.index(remainders[0]) == last_again + 1
    assert lengths == [n for n in warmup.spread_order(64) if n not in ladder]
    assert lengths[:4] == [48, 40, 24, 56]
    # ... each a one-token request of exactly that length, named as such.
    assert all(s.requests[0].max_tokens == 1 for s in remainders)
    assert len(remainders[0].requests[0].token_ids) == 48  # the ids, drawn when read
    assert [s.name for s in remainders[:2]] == [
        "prefill 48 tokens (chunk remainder)",
        "prefill 40 tokens (chunk remainder)",
    ]
    assert "9 prefill shapes, 7 measured again, 55 chunk remainders)" in plan.describe()
    # The fast level leaves them: a test leg or a quick start takes the merged
    # ladder alone, and the shipped default (full) carries the hazard's answer.
    fast = plan_warmup(dataclasses.replace(SMALL, every_prefill_length=True), "fast")
    assert not any(s.kind == "remainder" for s in fast.steps)
    assert "remainder" not in fast.describe()
    assert [s.kind for s in fast.steps] == [s.kind for s in plan_warmup(SMALL, "fast").steps]
    # Without the ask, the plan reads as before.
    before = plan_warmup(SMALL, "full")
    assert not any(s.kind == "remainder" for s in before.steps)
    assert "remainder" not in before.describe()


def test_a_fast_start_on_an_engine_that_asks_for_every_length_warns_that_the_hazard_stays(caplog):
    caplog.set_level(logging.WARNING, logger="smg_grpc_servicer.warmup")
    asks = dataclasses.replace(SMALL, every_prefill_length=True)
    plan = warmup.plan_for(warmup.WarmupSetting(True, "fast"), asks, knob="K")
    assert plan is not None and not any(s.kind == "remainder" for s in plan.steps)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and warned[0].startswith(
        "Planned warm-up at the fast level plans the length classes alone and leaves out the "
        "chunk remainders this engine asks for"
    )
    caplog.clear()
    # The full level runs them, and an engine that does not ask gets no warning at either level.
    assert warmup.plan_for(warmup.WarmupSetting(True, "full"), asks, knob="K") is not None
    assert warmup.plan_for(warmup.WarmupSetting(True, "fast"), SMALL, knob="K") is not None
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_the_remainders_cover_the_counts_a_chunked_prefills_last_chunk_can_take():
    # A 4,096-token chunk under an 8-request window: the class ladder visits
    # 54 of the 4,096 counts a step can hold; the last chunk of a longer
    # prefill is any of the rest (two of them hung an engine at first use).
    chunk = WarmupInputs(
        max_num_seqs=8, max_num_batched_tokens=4096, max_model_len=1 << 20, vocab_size=129280
    )
    ladder = {s.requests[0].prompt_tokens for s in plan_warmup(chunk).steps if s.kind == "prefill"}
    assert len(ladder) == 54 and not {3024, 1853} & ladder
    full = plan_warmup(dataclasses.replace(chunk, every_prefill_length=True))
    covered = {
        s.requests[0].prompt_tokens for s in full.steps if s.kind in ("prefill", "remainder")
    }
    assert covered == set(range(1, 4097))
    remainders = [s.requests[0].prompt_tokens for s in full.steps if s.kind == "remainder"]
    assert len(remainders) == 4096 - 54
    # The run's every prefix is a spread: its first 94 counts sit on a 32-token
    # pace, the first 31 of them on a 64-token one.
    assert all(n % 64 == 0 for n in remainders[:31]) and all(n % 32 == 0 for n in remainders[:94])
    assert full.prompt_tokens - plan_warmup(chunk).prompt_tokens == sum(range(1, 4097)) - sum(
        ladder
    )


def test_under_data_parallelism_every_rank_gets_every_remainder():
    two = plan_warmup(dataclasses.replace(SMALL, every_prefill_length=True, data_parallel_size=2))
    remainders = [s for s in two.steps if s.kind == "remainder"]
    assert len(remainders) == 55
    for step in remainders:
        assert [r.data_parallel_rank for r in step.requests] == [0, 1]
        assert len({r.prompt_tokens for r in step.requests}) == 1  # the same count on every rank
        assert len({r.token_ids for r in step.requests}) == 2  # its own tokens on each


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


class _Wire:
    """A wire that records what the pass sent, with a per-request delay and
    a failure rule."""

    def __init__(self, delay: float = 0.0, fail=lambda request: False):
        self.delay, self.fail = delay, fail
        self.requests: list = []
        self.timeouts: list[float] = []
        self.inflight = 0
        self.max_inflight = 0
        self.flushes = 0
        self.closed = False
        self._lock = threading.Lock()

    def generate(self, request, timeout):
        with self._lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            self.requests.append(request)
            self.timeouts.append(timeout)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail(request):
                raise RuntimeError(f"refused {request.request_id}")
        finally:
            with self._lock:
                self.inflight -= 1

    def flush_cache(self):
        self.flushes += 1
        return True

    def close(self):
        self.closed = True


def test_the_pass_runs_the_steps_in_order_with_a_steps_requests_together_and_one_line_each(caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.warmup")
    plan = plan_warmup(SMALL, "fast")
    wire = _Wire(delay=0.02)
    report = run_warmup(plan, wire, budget_secs=30)
    assert [r.request_id for r in wire.requests[:4]] == sorted(
        r.request_id for r in wire.requests[:4]
    )
    assert wire.max_inflight >= 2  # the staircase's requests went together, not one by one
    assert [r.request_id for r in wire.requests[4:]] == [
        step.requests[0].request_id for step in plan.steps[1:]
    ]
    assert report.covered == len(plan.steps) and report.errors == 0 and report.flushed is True
    assert not report.budget_hit and wire.flushes == 1
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Warm-up step ")]
    assert len(lines) == len(plan.steps)
    assert lines[0].startswith(f"Warm-up step 1/{len(plan.steps)} decode batch 4 down to 1")
    assert any(
        m.startswith("Planned warm-up starts (level fast:")
        for m in (r.getMessage() for r in caplog.records)
    )
    assert any(
        m.startswith("Planned warm-up over: ") for m in (r.getMessage() for r in caplog.records)
    )


def test_the_chunk_remainders_log_at_debug_with_a_progress_line_every_so_many_and_at_their_end(
    caplog, monkeypatch
):
    caplog.set_level(logging.DEBUG, logger="smg_grpc_servicer.warmup")
    monkeypatch.setattr(warmup, "REMAINDER_PROGRESS_EVERY", 4)
    plan = plan_warmup(dataclasses.replace(SMALL, every_prefill_length=True), "full")
    remainders = [s for s in plan.steps if s.kind == "remainder"]
    assert len(remainders) > 8 and len(remainders) % 4
    report = run_warmup(plan, _Wire(), budget_secs=30)
    assert report.covered == len(plan.steps) and report.errors == 0
    steps = [r for r in caplog.records if r.getMessage().startswith("Warm-up step ")]
    assert len(steps) == len(plan.steps)
    assert [r.levelno for r in steps if "(chunk remainder)" in r.getMessage()] == [
        logging.DEBUG
    ] * len(remainders)
    assert all(
        r.levelno == logging.INFO for r in steps if "(chunk remainder)" not in r.getMessage()
    )
    progress = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Warm-up chunk remainders: ")
    ]
    assert [int(m.split(" ")[3]) for m in progress] == [
        *range(4, len(remainders), 4),
        len(remainders),
    ]
    assert all(r.levelno == logging.INFO for r in caplog.records if r.getMessage() in progress)
    assert progress[-1].startswith(
        f"Warm-up chunk remainders: {len(remainders)} of {len(remainders)} done in "
    ) and progress[-1].endswith("0 failed) (every count up to the step used once)")

    # A pass that stops among the remainders (three failed steps in a row)
    # reports the ones it ran once more, failures counted, not as complete.
    caplog.clear()
    first = sum(len(s.requests) for s in plan.steps[: plan.steps.index(remainders[0])])
    seen: list = []
    wire = _Wire(fail=lambda request: seen.append(request) or len(seen) > first + 4)
    report = run_warmup(plan, wire, budget_secs=30)
    assert report.stopped and report.errors == 3
    progress = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Warm-up chunk remainders: ")
    ]
    assert [int(m.split(" ")[3]) for m in progress] == [4, 7]
    assert progress[-1].endswith("3 failed)") and "used once" not in progress[-1]


def test_the_budget_ends_the_pass_between_steps_and_the_report_says_where():
    plan = plan_warmup(SMALL, "fast")
    wire = _Wire(delay=0.05)
    report = run_warmup(plan, wire, budget_secs=0.12)
    assert report.budget_hit and 0 < len(report.results) < len(plan.steps)
    assert "the budget ended the pass before prefill" in report.summary()
    assert wire.flushes == 1  # what ran is still flushed


def test_a_failed_request_is_counted_and_the_pass_goes_on(caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.warmup")
    plan = plan_warmup(SMALL, "fast")
    wire = _Wire(fail=lambda request: len(request.token_ids) == 8 and request.max_tokens == 1)
    report = run_warmup(plan, wire, budget_secs=30)
    assert report.errors == 2  # the 8-token shape, first pass and second
    assert report.covered == len(plan.steps) - 2
    assert any(
        r.levelno == logging.WARNING
        and "prefill 8 tokens: 1 of 1 requests failed" in r.getMessage()
        for r in caplog.records
    )


def test_three_failed_steps_in_a_row_stop_the_pass():
    plan = plan_warmup(SMALL, "fast")
    wire = _Wire(fail=lambda request: True)
    report = run_warmup(plan, wire, budget_secs=30)
    assert len(report.results) == warmup.CONSECUTIVE_FAILED_STEPS
    assert report.stopped == f"{warmup.CONSECUTIVE_FAILED_STEPS} steps failed in a row"


def test_a_flush_the_engine_refuses_is_a_line_not_a_failure(caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.warmup")
    wire = _Wire()
    wire.flush_cache = lambda: (_ for _ in ()).throw(RuntimeError("UNIMPLEMENTED"))
    report = run_warmup(plan_warmup(SMALL, "fast"), wire, budget_secs=30)
    assert report.flushed is False and "was not flushed" in report.summary()
    # The end line says why, in the wire's words; so does the line at the refusal.
    assert report.flush_refusal == "UNIMPLEMENTED"
    assert report.summary().endswith("(the warm-up's blocks stay cached): UNIMPLEMENTED")
    assert "The warm-up's cache flush was not accepted: UNIMPLEMENTED" in caplog.text


@pytest.mark.parametrize("module", [vllm_warmup, sglang_warmup, tokenspeed_warmup])
def test_a_flush_the_servicer_answers_as_refused_raises_its_message(module):
    """A refusal the servicer answers (success false, with its reason) reaches
    the warm-up's end line like a raised status: the wire raises the message.
    An answer with no words stays a plain False; success stays True."""
    wire = module.GrpcWire.__new__(module.GrpcWire)
    refused = common_pb2.FlushCacheResponse(
        success=False, message="KV prefix cache reset refused; requests may be in flight"
    )
    wire._stub = SimpleNamespace(FlushCache=lambda request, timeout=None: refused)
    with pytest.raises(RuntimeError, match="requests may be in flight"):
        wire.flush_cache()
    flushed, refusal = warmup._flush(wire, logging.getLogger("test"))
    assert (flushed, refusal) == (False, "KV prefix cache reset refused; requests may be in flight")
    wire._stub = SimpleNamespace(
        FlushCache=lambda request, timeout=None: common_pb2.FlushCacheResponse(success=False)
    )
    assert wire.flush_cache() is False
    wire._stub = SimpleNamespace(
        FlushCache=lambda request, timeout=None: common_pb2.FlushCacheResponse(success=True)
    )
    assert wire.flush_cache() is True


def test_a_status_with_empty_details_is_worded_by_its_code():
    """A status whose details are empty would put its multi-line repr on the
    end line; its code's name is the one-line reason instead."""

    class _Status(Exception):
        def details(self):
            return ""

        def code(self):
            return SimpleNamespace(name="UNAVAILABLE")

        def __str__(self):
            return "<_InactiveRpcError of RPC that terminated with:\n\tstatus = UNAVAILABLE\n>"

    assert warmup._refusal(_Status()) == "UNAVAILABLE"
    wire = _Wire()
    wire.flush_cache = lambda: (_ for _ in ()).throw(_Status())
    flushed, refusal = warmup._flush(wire, logging.getLogger("test"))
    assert (flushed, refusal) == (False, "UNAVAILABLE") and "\n" not in refusal


def test_a_multi_line_refusal_is_worded_on_one_line():
    """Details with newlines (an engine's multi-line message forwarded by the
    servicer), and an exception text with newlines, read as one line on the
    end line; whitespace-only details fall through to the code's name."""

    class _Status(Exception):
        def __init__(self, text):
            super().__init__(text)
            self._text = text

        def details(self):
            return self._text

        def code(self):
            return SimpleNamespace(name="FAILED_PRECONDITION")

        def __str__(self):
            return "<_InactiveRpcError of RPC that terminated with:\n\tstatus = ...\n>"

    assert (
        warmup._refusal(_Status("engine busy:\n\tflush refused while 3 requests run\n"))
        == "engine busy: flush refused while 3 requests run"
    )
    assert warmup._refusal(_Status(" \n\t ")) == "FAILED_PRECONDITION"
    assert warmup._refusal(RuntimeError("line one\nline two")) == "line one line two"
    wire = _Wire()
    wire.flush_cache = lambda: (_ for _ in ()).throw(_Status("a\nb"))
    report = run_warmup(plan_warmup(SMALL, "fast"), wire, budget_secs=30)
    assert report.flush_refusal == "a b"
    assert "\n" not in report.summary() and report.summary().endswith(": a b")


def test_a_refused_flush_is_worded_by_the_status_details_not_its_repr():
    """A gRPC status' ``str`` runs over several lines (code, details, debug
    string); the end line carries its ``details()`` alone."""

    class _Status(RuntimeError):
        def __str__(self):
            return "<_InactiveRpcError of RPC that terminated with:\n\tstatus = UNIMPLEMENTED\n>"

        def details(self):
            return "FlushCache is not available through this servicer: the wire carries no flush"

    wire = _Wire()
    wire.flush_cache = lambda: (_ for _ in ()).throw(_Status())
    report = run_warmup(plan_warmup(SMALL, "fast"), wire, budget_secs=30)
    assert report.flushed is False
    assert report.flush_refusal == (
        "FlushCache is not available through this servicer: the wire carries no flush"
    )
    assert "\n" not in report.summary()
    assert report.summary().endswith(": " + report.flush_refusal)


def test_the_async_pass_words_a_refused_flush_the_same_way():
    class _Refusing:
        async def generate(self, request, timeout):
            return None

        async def flush_cache(self):
            raise RuntimeError("UNIMPLEMENTED")

        async def close(self):
            return None

    report = asyncio.run(
        warmup.run_warmup_async(plan_warmup(SMALL, "fast"), _Refusing(), budget_secs=30)
    )
    assert report.flushed is False and report.flush_refusal == "UNIMPLEMENTED"
    assert report.summary().endswith("(the warm-up's blocks stay cached): UNIMPLEMENTED")


def test_the_async_pass_matches_the_blocking_one():
    class AsyncWire:
        def __init__(self):
            self.requests = []
            self.flushes = 0

        async def generate(self, request, timeout):
            self.requests.append(request)
            await asyncio.sleep(0)

        async def flush_cache(self):
            self.flushes += 1
            return True

        async def close(self):
            return None

    plan = plan_warmup(SMALL, "fast")
    wire = AsyncWire()
    report = asyncio.run(warmup.run_warmup_async(plan, wire, budget_secs=30))
    assert report.covered == len(plan.steps) and wire.flushes == 1
    assert len(wire.requests) == sum(len(step.requests) for step in plan.steps)


def test_a_requests_timeout_is_the_callers_clamped_to_what_is_left_of_the_budget():
    plan = plan_warmup(SMALL, "fast")
    wire = _Wire(delay=0.01)
    run_warmup(plan, wire, budget_secs=5.0, request_timeout_secs=600.0)
    assert len(wire.timeouts) == sum(len(step.requests) for step in plan.steps)
    assert max(wire.timeouts) <= 5.0 and min(wire.timeouts) >= warmup.REQUEST_TIMEOUT_FLOOR_SECS
    assert wire.timeouts[0] > wire.timeouts[-1]  # the remainder shrinks as the pass runs
    wire = _Wire()
    run_warmup(plan, wire, budget_secs=5.0, request_timeout_secs=0.5)
    assert set(wire.timeouts) == {0.5}  # a shorter timeout of the caller's stands

    class AsyncWire:
        def __init__(self):
            self.timeouts = []

        async def generate(self, request, timeout):
            self.timeouts.append(timeout)

        async def flush_cache(self):
            return True

        async def close(self):
            return None

    wire = AsyncWire()
    asyncio.run(warmup.run_warmup_async(plan, wire, budget_secs=5.0, request_timeout_secs=600.0))
    assert max(wire.timeouts) <= 5.0 and min(wire.timeouts) >= warmup.REQUEST_TIMEOUT_FLOOR_SECS


def test_a_hanging_request_cannot_hold_the_pass_past_the_budget(monkeypatch):
    monkeypatch.setattr(warmup, "REQUEST_TIMEOUT_FLOOR_SECS", 0.05)

    class Hanging(_Wire):
        def generate(self, request, timeout):
            super().generate(request, timeout)
            time.sleep(timeout)  # the engine never answers; the wire's deadline fires
            raise TimeoutError(f"deadline after {timeout:.2f} s")

    plan = plan_warmup(SMALL, "fast")
    started = time.monotonic()
    report = run_warmup(plan, Hanging(), budget_secs=0.3, request_timeout_secs=600.0)
    elapsed = time.monotonic() - started
    assert report.budget_hit and report.errors >= 1 and report.covered == 0
    assert elapsed < 1.0  # the budget, the floor and some slack; never the 600 s timeout
    assert "the budget ended the pass" in report.summary()


def test_a_thread_the_host_refuses_fails_that_request_and_the_pass_goes_on(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.warmup")
    real_submit = futures.ThreadPoolExecutor.submit
    granted = {"left": 2}

    def submit(self, fn, *args, **kwargs):
        if granted["left"] == 0:
            raise RuntimeError("can't start new thread")
        granted["left"] -= 1
        return real_submit(self, fn, *args, **kwargs)

    monkeypatch.setattr(futures.ThreadPoolExecutor, "submit", submit)
    plan = plan_warmup(SMALL, "fast")  # the staircase sends four requests together
    wire = _Wire()
    report = run_warmup(plan, wire, budget_secs=30)
    assert report.results[0].errors == 2 and report.covered == len(plan.steps) - 1
    assert len(wire.requests) == 2 + len(plan.steps) - 1
    assert any(
        "decode batch 4 down to 1 (4 requests): 2 of 4 requests failed" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# The wires, against an in-process stand-in for each engine's service
# ---------------------------------------------------------------------------


class _Seen:
    """What a stand-in engine saw: (prompt length, max tokens) per request,
    the flushes, the largest number of requests in flight at once."""

    def __init__(self):
        self.requests: list[tuple[int, int]] = []
        self.ranks: list[int | None] = []
        self.flushes = 0
        self.inflight = 0
        self.max_inflight = 0
        self.lock = threading.Lock()

    def enter(self, prompt_len, max_tokens, rank=None):
        with self.lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            self.requests.append((prompt_len, max_tokens))
            self.ranks.append(rank)

    def leave(self):
        with self.lock:
            self.inflight -= 1


class _VllmEngine(vllm_grpc.VllmEngineServicer):
    def __init__(self, seen):
        self.seen = seen

    def Generate(self, request, context):
        assert request.HasField("tokenized") and request.sampling_params.ignore_eos
        assert request.sampling_params.temperature == 0.0
        rank = request.data_parallel_rank if request.HasField("data_parallel_rank") else None
        self.seen.enter(len(request.tokenized.input_ids), request.sampling_params.max_tokens, rank)
        time.sleep(0.01)
        self.seen.leave()
        yield vllm_pb.GenerateResponse(complete=vllm_pb.GenerateComplete(finish_reason="length"))

    def FlushCache(self, request, context):
        self.seen.flushes += 1
        return common_pb2.FlushCacheResponse(success=True)


class _SglangEngine(sgl_grpc.SglangSchedulerServicer):
    def __init__(self, seen):
        self.seen = seen

    def Generate(self, request, context):
        params = request.sampling_params
        assert params.ignore_eos and params.top_k == 1 and params.top_p == 1.0 and params.n == 1
        self.seen.enter(len(request.tokenized.input_ids), params.max_new_tokens)
        time.sleep(0.01)
        self.seen.leave()
        yield sgl_pb.GenerateResponse(complete=sgl_pb.GenerateComplete())

    def FlushCache(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "no cache flush on this wire")


class _TokenSpeedEngine(ts_grpc.TokenSpeedSchedulerServicer):
    def __init__(self, seen):
        self.seen = seen

    def Generate(self, request, context):
        assert request.sampling_params.ignore_eos and request.sampling_params.temperature == 0.0
        self.seen.enter(len(request.tokenized.input_ids), request.sampling_params.max_new_tokens)
        time.sleep(0.01)
        self.seen.leave()
        yield ts_pb.GenerateResponse(complete=ts_pb.GenerateComplete())

    def FlushCache(self, request, context):
        self.seen.flushes += 1
        return common_pb2.FlushCacheResponse(success=True)


def _serve(add, servicer):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    add(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


@pytest.mark.parametrize(
    "add, engine, wire_of, flushable",
    [
        (vllm_grpc.add_VllmEngineServicer_to_server, _VllmEngine, vllm_warmup.warmup_wire, True),
        (
            sgl_grpc.add_SglangSchedulerServicer_to_server,
            _SglangEngine,
            sglang_warmup.warmup_wire,
            False,
        ),
        (
            ts_grpc.add_TokenSpeedSchedulerServicer_to_server,
            _TokenSpeedEngine,
            tokenspeed_warmup.warmup_wire,
            True,
        ),
    ],
    ids=["vllm", "sglang", "tokenspeed"],
)
def test_each_wire_drives_the_plan_through_the_engines_own_service(add, engine, wire_of, flushable):
    seen = _Seen()
    server, target = _serve(add, engine(seen))
    try:
        plan = plan_warmup(SMALL, "fast")
        wire = wire_of(target)
        try:
            report = run_warmup(plan, wire, budget_secs=30, request_timeout_secs=10)
        finally:
            wire.close()
    finally:
        server.stop(0)
    assert report.covered == len(plan.steps) and report.errors == 0
    expected = [(len(r.token_ids), r.max_tokens) for step in plan.steps for r in step.requests]
    assert sorted(seen.requests) == sorted(expected)
    assert seen.max_inflight >= 2  # the staircase reached the engine together, not one by one
    assert report.flushed is flushable and seen.flushes == (1 if flushable else 0)
    # A refusal over the wire reaches the end line as the status' details.
    assert report.flush_refusal == (None if flushable else "no cache flush on this wire")


def test_the_vllm_wire_pins_each_request_to_its_rank_over_the_servicers_port():
    seen = _Seen()
    server, target = _serve(vllm_grpc.add_VllmEngineServicer_to_server, _VllmEngine(seen))
    try:
        wire = vllm_warmup.warmup_wire(target)
        try:
            two = plan_warmup(dataclasses.replace(SMALL, data_parallel_size=2), "fast")
            report = run_warmup(two, wire, budget_secs=30, request_timeout_secs=10)
            pinned = len(seen.requests)
            run_warmup(plan_warmup(SMALL, "fast"), wire, budget_secs=30, request_timeout_secs=10)
        finally:
            wire.close()
    finally:
        server.stop(0)
    assert report.covered == len(two.steps) and report.errors == 0
    one_ranks_shapes = sorted(
        (len(r.token_ids), r.max_tokens)
        for s in plan_warmup(SMALL, "fast").steps
        for r in s.requests
    )
    for rank in (0, 1):  # every rank saw every shape
        seen_on_rank = [
            shape for shape, r in zip(seen.requests[:pinned], seen.ranks[:pinned]) if r == rank
        ]
        assert sorted(seen_on_rank) == one_ranks_shapes
    assert set(seen.ranks[pinned:]) == {None}  # one rank: the field stays unset


def test_a_wire_over_a_borrowed_channel_leaves_it_open():
    seen = _Seen()
    server, target = _serve(sgl_grpc.add_SglangSchedulerServicer_to_server, _SglangEngine(seen))
    try:
        channel = grpc.insecure_channel(target)
        wire = sglang_warmup.warmup_wire("", channel=channel)
        wire.generate(warmup.WarmupRequest("r", 3, 2, seed=0, vocab_size=100), 10)
        wire.close()
        wire.generate(warmup.WarmupRequest("r2", 4, 1, seed=1, vocab_size=100), 10)  # still open
        channel.close()
    finally:
        server.stop(0)
    assert seen.requests == [(3, 2), (4, 1)]


# ---------------------------------------------------------------------------
# The engines' inputs
# ---------------------------------------------------------------------------


def test_the_vllm_inputs_come_from_the_resolved_configuration():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="m", max_model_len=4096, get_vocab_size=lambda: 32000, runner_type="generate"
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=2048),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[8, 1, 4, 2, 16]),
    )
    inputs = vllm_warmup.warmup_inputs_from_config(config)
    assert (
        inputs.max_num_seqs,
        inputs.max_num_batched_tokens,
        inputs.max_model_len,
        inputs.vocab_size,
    ) == (16, 2048, 4096, 32000)
    assert inputs.capture_sizes == (1, 2, 4, 8, 16)
    assert inputs.data_parallel_size == 1  # no parallel config: one rank
    config.parallel_config = SimpleNamespace(data_parallel_size=4)
    assert vllm_warmup.warmup_inputs_from_config(config).data_parallel_size == 4
    config.model_config.runner_type = "pooling"
    assert vllm_warmup.warmup_inputs_from_config(config) is None


def test_the_sglang_inputs_come_from_the_server_arguments_and_the_facts():
    args = SimpleNamespace(max_running_requests=64, chunked_prefill_size=8192, context_length=None)
    facts = {"is_generation": True, "max_context_length": 32768, "vocab_size": 151936}
    inputs = sglang_warmup.warmup_inputs(args, facts)
    assert (
        inputs.max_num_seqs,
        inputs.max_num_batched_tokens,
        inputs.max_model_len,
        inputs.vocab_size,
    ) == (64, 8192, 32768, 151936)
    # Chunking off: the prefill budget is the step; none: the engine's default.
    assert (
        sglang_warmup.warmup_inputs(
            SimpleNamespace(chunked_prefill_size=-1, max_prefill_tokens=4096), facts
        ).max_num_batched_tokens
        == 4096
    )
    assert (
        sglang_warmup.warmup_inputs(SimpleNamespace(), facts).max_num_batched_tokens
        == sglang_warmup.DEFAULT_STEP_TOKENS
    )
    assert sglang_warmup.warmup_inputs(SimpleNamespace(), facts).max_num_seqs == 0
    assert sglang_warmup.warmup_inputs(args, {**facts, "is_generation": False}) is None
    assert (
        sglang_warmup.warmup_inputs(SimpleNamespace(disaggregation_mode="prefill"), facts) is None
    )
    assert sglang_warmup.warmup_inputs(SimpleNamespace(language_only=True), facts) is None


def test_the_tokenspeed_inputs_come_from_the_server_arguments_and_the_facts():
    args = SimpleNamespace(
        max_num_seqs=32, chunked_prefill_size=None, max_prefill_tokens=8192, max_model_len=16384
    )
    facts = {"max_context_length": 32768, "vocab_size": 128256}
    inputs = tokenspeed_warmup.warmup_inputs(args, facts)
    assert (
        inputs.max_num_seqs,
        inputs.max_num_batched_tokens,
        inputs.max_model_len,
        inputs.vocab_size,
    ) == (32, 8192, 16384, 128256)
    assert tokenspeed_warmup.warmup_inputs(SimpleNamespace(), facts).max_model_len == 32768
    assert (
        tokenspeed_warmup.warmup_inputs(SimpleNamespace(), facts).max_num_batched_tokens
        == tokenspeed_warmup.DEFAULT_STEP_TOKENS
    )
    assert (
        tokenspeed_warmup.warmup_inputs(SimpleNamespace(disaggregation_mode="decode"), facts)
        is None
    )


def test_the_sglang_and_tokenspeed_inputs_say_when_the_pass_spreads_over_ranks(caplog):
    caplog.set_level(logging.WARNING, logger="smg_grpc_servicer")
    facts = {"is_generation": True, "max_context_length": 32768, "vocab_size": 151936}
    sglang = sglang_warmup.warmup_inputs(SimpleNamespace(dp_size=4, max_running_requests=64), facts)
    tokenspeed = tokenspeed_warmup.warmup_inputs(
        SimpleNamespace(mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=2))), facts
    )
    # One rank's plan: the servicer routes these engines' requests by load.
    assert sglang.data_parallel_size == 1 and tokenspeed.data_parallel_size == 1
    spread = [r.getMessage() for r in caplog.records if "reaches one rank" in r.getMessage()]
    assert [line.split(" scheduler ranks")[0] for line in spread] == [
        "The engine runs 4",
        "The engine runs 2",
    ]
    # TokenSpeed's remainders answer a hazard, and reach one rank each too.
    assert "chunk remainders, when the plan runs them (the full level), reach one rank" in spread[1]
    assert "chunk remainders" not in spread[0]
    caplog.clear()
    sglang_warmup.warmup_inputs(SimpleNamespace(max_running_requests=64), facts)
    tokenspeed_warmup.warmup_inputs(SimpleNamespace(), facts)
    assert not caplog.records


def test_only_the_tokenspeed_inputs_ask_for_every_prefill_length():
    facts = {"is_generation": True, "max_context_length": 32768, "vocab_size": 151936}
    args = SimpleNamespace(max_num_seqs=8, chunked_prefill_size=4096)
    assert tokenspeed_warmup.warmup_inputs(args, facts).every_prefill_length is True
    assert sglang_warmup.warmup_inputs(args, facts).every_prefill_length is False
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="m", max_model_len=4096, get_vocab_size=lambda: 32000, runner_type="generate"
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=2048),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[8]),
    )
    assert vllm_warmup.warmup_inputs_from_config(config).every_prefill_length is False
    # The plan the launcher runs for this engine covers every count of its chunk.
    plan = plan_warmup(tokenspeed_warmup.warmup_inputs(args, facts))
    assert {
        s.requests[0].prompt_tokens for s in plan.steps if s.kind in ("prefill", "remainder")
    } == set(range(1, 4097))


# ---------------------------------------------------------------------------
# The lifecycle: readiness held until the pass ran
# ---------------------------------------------------------------------------


class _ServerOf:
    def __init__(self):
        self.engine_ready = False
        self.kv_relay_subscribed = True
        self.last_error = None
        self.running = True
        self.address = "0.0.0.0:50051"
        self.events: list[str] = []

    def set_serving(self, serving: bool) -> None:
        self.events.append(f"serving:{serving}")

    def set_ready(self, ready: bool) -> None:
        self.events.append(f"ready:{ready}")

    def stop(self, timeout: float) -> None:
        self.events.append("stop")


class _EngineOf:
    def __init__(self):
        self.code = None
        self.events: list[str] = []

    def poll(self):
        return self.code

    def terminate(self):
        self.events.append("terminate")
        self.code = 0

    def kill(self):
        self.events.append("kill")
        self.code = -9

    def wait(self, timeout=None):
        return self.code

    def kill_descendants(self):
        return 0


def _supervised(
    warmup_fn,
    *,
    ready_after: float = 0.03,
    stop_after: float = 0.25,
    relay_joins_after: float | None = None,
    server: _ServerOf | None = None,
):
    """``supervise`` over a fake server and engine: the engine connects
    ``ready_after`` seconds in; with ``relay_joins_after`` the server's
    KV-event relay reports its subscription only that long after the
    connect (``None``: it is subscribed from the start, or the server says
    nothing about it)."""
    server, engine = server or _ServerOf(), _EngineOf()
    stop_event_holder = {}

    async def run():
        stop_event = asyncio.Event()
        stop_event_holder["event"] = stop_event
        task = asyncio.create_task(
            rust_lifecycle.supervise(
                server,
                engine,
                drain_secs=0.0,
                stop_timeout=0.1,
                stop_event=stop_event,
                poll_secs=0.005,
                warmup=warmup_fn,
            )
        )
        await asyncio.sleep(ready_after)
        server.engine_ready = True
        if relay_joins_after is not None:
            await asyncio.sleep(relay_joins_after)
            server.kv_relay_subscribed = True
        await asyncio.sleep(stop_after)
        stop_event.set()
        return await task

    code = asyncio.run(run())
    return code, server, engine


def test_the_servicer_serves_only_after_the_planned_warm_up_ran(caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.rust_lifecycle")
    ran = threading.Event()

    def warm():
        time.sleep(0.05)
        ran.set()
        return warmup.WarmupReport(plan_warmup(SMALL, "fast"), secs=0.05)

    code, server, engine = _supervised(warm)
    assert code == 0 and ran.is_set()
    # The release is the health-only setter: admission was never closed.
    assert server.events == ["ready:True", "serving:False", "stop"]
    assert engine.events == ["terminate"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("the planned warm-up runs before the servicer is SERVING" in m for m in messages)
    assert any(
        m.startswith("Warm-up over (0 of 17 steps covered")
        and m.endswith("the servicer is SERVING")
        for m in messages
    )


def test_a_warm_up_that_fails_still_lets_the_servicer_serve(caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.rust_lifecycle")

    def warm():
        raise RuntimeError("the wire refused")

    code, server, _ = _supervised(warm)
    assert code == 0
    assert server.events == ["ready:True", "serving:False", "stop"]
    assert any(
        r.levelno == logging.ERROR
        and "The planned warm-up failed; the servicer is SERVING without it" in r.getMessage()
        for r in caplog.records
    )


def test_without_a_warm_up_the_servicer_serves_at_the_handshake(caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.rust_lifecycle")
    code, server, _ = _supervised(None, stop_after=0.05)
    assert code == 0 and server.events == ["serving:False", "stop"]
    assert any(
        r.getMessage() == "Engine connected; the servicer is SERVING" for r in caplog.records
    )


def _warm_up_noting(server_holder: dict, seen: list):
    """A warm-up that records whether the relay was subscribed when it ran."""

    def warm():
        seen.append(server_holder["server"].kv_relay_subscribed)
        return warmup.WarmupReport(plan_warmup(SMALL, "fast"), secs=0.0)

    return warm


def test_the_planned_warm_up_waits_for_the_relay_to_join_the_publisher(caplog):
    """The engine connects before the KV-event relay's subscription reached
    its publisher: the warm-up (whose batches are the engine's first) starts
    only once the server reports the subscription, so the relay's history
    begins with them; readiness stays held meanwhile and flips after the
    pass as before."""
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.rust_lifecycle")
    server = _ServerOf()
    server.kv_relay_subscribed = False
    holder, seen = {"server": server}, []
    code, server, _ = _supervised(
        _warm_up_noting(holder, seen), relay_joins_after=0.1, server=server
    )
    assert code == 0
    assert seen == [True], "the warm-up ran before the relay joined"
    assert server.events == ["ready:True", "serving:False", "stop"]
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("KV event relay has not joined") and "waits" in m for m in messages)
    assert any(
        m.startswith("KV event relay joined the engine's publisher")
        and m.endswith("the planned warm-up starts")
        for m in messages
    )


def test_the_planned_warm_up_starts_without_the_relay_after_the_grace(monkeypatch, caplog):
    """A relay that never joins (its publisher never comes up) holds the
    warm-up for the grace and no longer: the pass runs, the servicer serves,
    one warning says what the relay will miss."""
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.rust_lifecycle")
    monkeypatch.setattr(rust_lifecycle, "RELAY_JOIN_GRACE_SECS", 0.05)
    server = _ServerOf()
    server.kv_relay_subscribed = False
    holder, seen = {"server": server}, []
    code, server, _ = _supervised(_warm_up_noting(holder, seen), server=server)
    assert code == 0
    assert seen == [False]
    assert server.events == ["ready:True", "serving:False", "stop"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and warnings[0].startswith("KV event relay has not joined")
    assert "the planned warm-up starts without it" in warnings[0]


def test_a_server_that_says_nothing_about_its_relay_warms_up_at_the_handshake(caplog):
    """An older binding without ``kv_relay_subscribed`` is not held."""
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer.rust_lifecycle")
    server = _ServerOf()
    del server.kv_relay_subscribed
    ran = threading.Event()

    def warm():
        ran.set()
        return warmup.WarmupReport(plan_warmup(SMALL, "fast"), secs=0.0)

    code, server, _ = _supervised(warm, server=server)
    assert code == 0 and ran.is_set()
    assert server.events == ["ready:True", "serving:False", "stop"]
    assert not any("KV event relay" in r.getMessage() for r in caplog.records)


def test_the_hold_flips_health_alone_and_falls_back_on_an_older_binding():
    """``set_serving(False)`` is the drain switch and closes admission, which
    would refuse the warm-up's own requests: the hold uses ``set_ready`` where
    the server has it; an older binding without one gets ``set_serving``."""
    server = _ServerOf()
    rust_lifecycle.hold_readiness(server, False)
    rust_lifecycle.hold_readiness(server, True)
    assert server.events == ["ready:False", "ready:True"]

    class Older:
        def __init__(self):
            self.events = []

        def set_serving(self, serving):
            self.events.append(f"serving:{serving}")

    older = Older()
    rust_lifecycle.hold_readiness(older, False)
    assert older.events == ["serving:False"]


def test_the_real_server_classes_carry_the_health_only_setter():
    """The Rust servers of every engine expose ``set_ready`` beside
    ``set_serving`` (the semantics are pinned by the servers' own tests: held
    = NOT_SERVING with admission open; drained = admission closed)."""
    servicer = pytest.importorskip("smg.servicer")
    for name in ("VllmGrpcServer", "SglangGrpcServer", "TokenSpeedGrpcServer"):
        cls = getattr(servicer, name)
        assert callable(getattr(cls, "set_ready")) and callable(getattr(cls, "set_serving"))


def test_the_launchers_run_warmup_dials_the_servers_own_listener(monkeypatch):
    class Launcher(rust_lifecycle.RustLauncher):
        warmup_env = "SMG_X_SERVICER_WARMUP"

        def __init__(self):
            self.targets = []

        def warmup_wire(self, target):
            self.targets.append(target)
            return _Wire()

    launcher = Launcher()
    report = launcher.run_warmup(
        plan_warmup(SMALL, "fast"), warmup.WarmupSetting(True, "fast", 5.0), _ServerOf()
    )
    assert launcher.targets == ["127.0.0.1:50051"]
    assert report.covered == 17
    assert rust_lifecycle.RustLauncher().warmup_inputs(None, {}) is None


# ---------------------------------------------------------------------------
# The Python servicer: the health service holds while the pass runs
# ---------------------------------------------------------------------------


class _AsyncLlm:
    def __init__(self):
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="m", max_model_len=256, get_vocab_size=lambda: 32000, runner_type="generate"
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4]),
        )
        self.requests: list[tuple[int, int]] = []
        self.pins: list[dict] = []
        self.resets = 0

    async def generate(self, prompt, params, request_id, **kwargs):
        self.requests.append((len(prompt.prompt_token_ids), params.max_tokens))
        self.pins.append(kwargs)
        await asyncio.sleep(0.002)
        yield SimpleNamespace(request_id=request_id)

    async def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        self.resets += 1
        return True


@pytest.fixture
def fake_vllm(monkeypatch):
    """A ``vllm`` whose sampling classes record their arguments: the
    in-process wire imports nothing else of it."""
    module = types.ModuleType("vllm")
    module.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    module.TokensPrompt = lambda prompt_token_ids: SimpleNamespace(
        prompt_token_ids=prompt_token_ids
    )
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(vllm_warmup._running, "task", None)
    return module


def test_the_python_servicers_pass_holds_the_health_until_it_is_over(fake_vllm, caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer")
    engine = _AsyncLlm()
    seen = {}

    async def run():
        task = vllm_warmup.start_engine_warmup(engine, setting_text="fast:10")
        assert task is not None
        seen["during"] = vllm_warmup.warmup_in_progress()
        report = await task
        seen["after"] = vllm_warmup.warmup_in_progress()
        return report

    report = asyncio.run(run())
    assert seen == {"during": True, "after": False}
    plan = plan_warmup(vllm_warmup.warmup_inputs_from_config(engine.vllm_config), "fast")
    assert report.covered == len(plan.steps) and engine.resets == 1
    assert sorted(engine.requests) == sorted(
        (len(r.token_ids), r.max_tokens) for s in plan.steps for r in s.requests
    )
    assert all(pin == {} for pin in engine.pins)  # one rank: the client's routing is left alone


def test_the_python_servicers_pass_pins_each_request_to_its_rank(fake_vllm):
    engine = _AsyncLlm()
    engine.vllm_config.parallel_config = SimpleNamespace(data_parallel_size=2)

    async def run():
        return await vllm_warmup.start_engine_warmup(engine, setting_text="fast:10")

    report = asyncio.run(run())
    inputs = vllm_warmup.warmup_inputs_from_config(engine.vllm_config)
    assert inputs.data_parallel_size == 2
    assert report.covered == len(plan_warmup(inputs, "fast").steps) and report.errors == 0
    one_ranks_shapes = sorted(
        (len(r.token_ids), r.max_tokens)
        for s in plan_warmup(dataclasses.replace(inputs, data_parallel_size=1), "fast").steps
        for r in s.requests
    )
    for rank in (0, 1):  # every rank saw every shape, by the engine client's own argument
        seen_on_rank = [
            shape
            for shape, pin in zip(engine.requests, engine.pins)
            if pin.get("data_parallel_rank") == rank
        ]
        assert sorted(seen_on_rank) == one_ranks_shapes


def test_the_python_servicer_skips_the_pass_when_the_knob_is_off_or_bad(fake_vllm, caplog):
    caplog.set_level(logging.INFO, logger="smg_grpc_servicer")
    engine = _AsyncLlm()

    async def run():
        assert vllm_warmup.start_engine_warmup(engine, setting_text="off") is None
        assert vllm_warmup.start_engine_warmup(engine, setting_text="warm") is None
        assert vllm_warmup.start_engine_warmup(SimpleNamespace(), setting_text=None) is None

    asyncio.run(run())
    assert vllm_warmup.start_engine_warmup(engine, setting_text=None) is None  # no loop runs here
    messages = [r.getMessage() for r in caplog.records]
    assert any("off (SMG_VLLM_SERVICER_WARMUP=off)" in m for m in messages)
    assert any("SMG_VLLM_SERVICER_WARMUP='warm'" in m for m in messages)
    assert any("could not be read" in m for m in messages)
    assert any("No event loop runs here" in m for m in messages)


def _health_servicer_module():
    """The vLLM health servicer, loaded with a stand-in for vLLM's logger."""
    if "vllm.logger" not in sys.modules:
        vllm = types.ModuleType("vllm")
        vllm_logger = types.ModuleType("vllm.logger")
        vllm_logger.init_logger = logging.getLogger
        vllm.logger = vllm_logger
        sys.modules.setdefault("vllm", vllm)
        sys.modules["vllm.logger"] = vllm_logger
    path = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "health_servicer.py"
    spec = importlib.util.spec_from_file_location("warmup_health_servicer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_health_service_answers_not_serving_while_the_pass_runs(monkeypatch):
    module = _health_servicer_module()

    class Engine:
        async def check_health(self):
            return None

    servicer = module.VllmHealthServicer(Engine())

    async def run():
        gate = asyncio.Event()

        async def pass_():
            await gate.wait()

        task = asyncio.create_task(pass_())
        monkeypatch.setitem(vllm_warmup._running, "task", task)
        during = await servicer.engine_status(servicer.VLLM_SERVICE)
        gate.set()
        await task
        after = await servicer.engine_status(servicer.VLLM_SERVICE)
        return during, after

    assert asyncio.run(run()) == (module.NOT_SERVING, module.SERVING)
