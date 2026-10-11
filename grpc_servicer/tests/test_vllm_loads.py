"""The vLLM servicer's load bookkeeping: queued token-work, generation throughput
and the hit rate from what it forwards and streams (no engine needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("smg_grpc_proto")


@pytest.fixture(scope="module")
def loads_mod():
    """Load loads.py by path: the vllm package __init__ imports the engine."""
    path = Path(__file__).resolve().parent.parent / "smg_grpc_servicer" / "vllm" / "loads.py"
    spec = importlib.util.spec_from_file_location("vllm_loads_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_queued_token_work_is_the_youngest_waiting_prompts_discounted_by_the_hit_rate(loads_mod):
    tracker = loads_mod.LoadTracker(clock=Clock())
    assert tracker.estimate(5) == loads_mod.LoadEstimate()

    # Three in flight, none started: the engine says two are waiting, so the
    # oldest is in prefill and the two youngest are queued work.
    tracker.submitted("a", 1000)
    tracker.submitted("b", 300)
    tracker.submitted("c", 200)
    assert tracker.estimate(2).queued_token_work == 500
    assert tracker.estimate(0).queued_token_work == 0
    assert tracker.estimate(10).queued_token_work == 1500, "never more than we hold"

    # The first output of `a`: half its prompt was cached; queued prompts are
    # discounted by that rate.
    tracker.first_output("a", 1000, 500)
    assert tracker.pending() == 2
    estimate = tracker.estimate(2)
    assert estimate.cache_hit_rate == 0.5
    assert estimate.queued_token_work == 250

    tracker.finished("b")  # ended before it started
    assert tracker.estimate(2).queued_token_work == 100
    tracker.first_output("c", 200, 200)
    assert tracker.pending() == 0
    assert tracker.estimate(3).queued_token_work == 0
    assert tracker.estimate(0).cache_hit_rate == pytest.approx(700 / 1200)


def test_throughput_counts_tokens_inside_the_window_only(loads_mod):
    clock = Clock()
    tracker = loads_mod.LoadTracker(clock=clock)
    tracker.generated(1000)
    clock.now += 1.0
    tracker.generated(3000)
    assert tracker.estimate(0).gen_throughput == 2000.0
    clock.now += 1.5  # the first sample falls out of the two-second window
    assert tracker.estimate(0).gen_throughput == 1500.0
    clock.now += 2.0
    assert tracker.estimate(0).gen_throughput == 0.0
    tracker.generated(0)
    assert tracker.estimate(0).gen_throughput == 0.0


def test_the_hit_rate_averages_recent_first_outputs_and_ignores_empty_prompts(loads_mod):
    tracker = loads_mod.LoadTracker(clock=Clock())
    tracker.submitted("x", 0)
    tracker.first_output("x", 0, 0)
    assert tracker.estimate(0).cache_hit_rate == 0.0
    for index in range(loads_mod.HIT_RATE_SAMPLES):
        tracker.first_output(f"old-{index}", 100, 0)
    assert tracker.estimate(0).cache_hit_rate == 0.0
    for index in range(loads_mod.HIT_RATE_SAMPLES):
        tracker.first_output(f"new-{index}", 100, 100)
    assert tracker.estimate(0).cache_hit_rate == 1.0, "the old samples aged out"
    tracker.first_output("odd", 10, 50)
    assert tracker.estimate(0).cache_hit_rate <= 1.0


def test_scheduler_load_fields_carry_vllms_counts_and_the_estimate(loads_mod):
    estimate = loads_mod.LoadEstimate(
        queued_token_work=640, gen_throughput=1234.5, cache_hit_rate=0.25
    )
    fields = loads_mod.scheduler_load_fields(
        3, 2, 0.4, estimate, max_total_num_tokens=10_000, max_running_requests=64
    )
    assert fields == {
        "dp_rank": 0,
        "num_running_reqs": 3,
        "num_waiting_reqs": 2,
        "num_waiting_uncached_tokens": 640,
        "num_total_reqs": 5,
        "token_usage": 0.4,
        "utilization": 0.4,
        "gen_throughput": 1234.5,
        "cache_hit_rate": 0.25,
        "max_total_num_tokens": 10_000,
        "num_used_tokens": 4_000,
        "max_running_requests": 64,
    }
    # Unknown capacity figures stay unset rather than zero-filled; a negative
    # KV usage (a stats race) reads as empty.
    bare = loads_mod.scheduler_load_fields(0, 0, -0.1, loads_mod.LoadEstimate())
    assert "max_total_num_tokens" not in bare and "num_used_tokens" not in bare
    assert bare["token_usage"] == 0.0

    from smg_grpc_proto.generated import vllm_engine_pb2

    load = vllm_engine_pb2.SchedulerLoad(**fields)
    assert (load.num_waiting_uncached_tokens, load.gen_throughput) == (640, 1234.5)


def test_the_load_record_agrees_with_the_get_loads_fields(loads_mod):
    estimate = loads_mod.LoadEstimate(
        queued_token_work=1_200, gen_throughput=333.0, cache_hit_rate=0.4
    )
    fields = loads_mod.scheduler_load_fields(
        3, 2, 0.125, estimate, max_total_num_tokens=8_000, max_running_requests=64
    )
    record = loads_mod.load_record(
        3, 2, 0.125, estimate, max_total_num_tokens=8_000, max_running_requests=64
    )
    assert record.running_requests == fields["num_running_reqs"] == 3
    assert record.waiting_requests == fields["num_waiting_reqs"] == 2
    assert record.token_usage == fields["token_usage"] == 0.125
    assert record.gen_throughput == fields["gen_throughput"] == 333.0
    assert record.cache_hit_rate == fields["cache_hit_rate"] == 0.4
    assert record.waiting_uncached_tokens == fields["num_waiting_uncached_tokens"] == 1_200
    assert record.num_used_tokens == fields["num_used_tokens"] == 1_000
    assert record.max_total_num_tokens == fields["max_total_num_tokens"] == 8_000
    assert record.max_running_requests == fields["max_running_requests"] == 64
    assert not record.HasField("memory")


def test_the_load_record_before_the_first_step_is_zero_filled_with_the_capacity_unknown(loads_mod):
    record = loads_mod.load_record(0, 0, 0.0, loads_mod.LoadEstimate())
    assert (record.running_requests, record.waiting_requests, record.token_usage) == (0, 0, 0.0)
    assert record.HasField("waiting_uncached_tokens") and record.waiting_uncached_tokens == 0
    assert record.max_total_num_tokens == 0 and record.max_running_requests == 0


def test_the_load_record_of_another_rank_carries_its_own_counts_and_no_estimate(loads_mod):
    fields = loads_mod.scheduler_load_fields(4, 1, 0.5, None, dp_rank=1, max_total_num_tokens=8_000)
    assert fields["dp_rank"] == 1
    assert not {"num_waiting_uncached_tokens", "gen_throughput", "cache_hit_rate"} & fields.keys()
    record = loads_mod.load_record(4, 1, 0.5, None, dp_rank=1, max_total_num_tokens=8_000)
    assert (record.running_requests, record.waiting_requests, record.token_usage) == (4, 1, 0.5)
    assert record.num_used_tokens == 4_000
    # The estimate is the servicer's and rides the first rank's record: absent here, not 0.
    assert not record.HasField("waiting_uncached_tokens")
    assert (record.gen_throughput, record.cache_hit_rate) == (0.0, 0.0)
