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
    assert tracker.estimate(5, rank=0) == loads_mod.LoadEstimate()

    # Three in flight, none started: the engine says two are waiting, so the
    # oldest is in prefill and the two youngest are queued work.
    tracker.submitted("a", 1000, rank=0)
    tracker.submitted("b", 300, rank=0)
    tracker.submitted("c", 200, rank=0)
    assert tracker.estimate(2, rank=0).queued_token_work == 500
    assert tracker.estimate(0, rank=0).queued_token_work == 0
    assert tracker.estimate(10, rank=0).queued_token_work == 1500, "never more than we hold"

    # The first output of `a`: half its prompt was cached; queued prompts are
    # discounted by that rate.
    tracker.first_output("a", 1000, 500, rank=0)
    assert tracker.pending() == 2
    estimate = tracker.estimate(2, rank=0)
    assert estimate.cache_hit_rate == 0.5
    assert estimate.queued_token_work == 250

    tracker.finished("b")  # ended before it started
    assert tracker.estimate(2, rank=0).queued_token_work == 100
    tracker.first_output("c", 200, 200, rank=0)
    assert tracker.pending() == 0
    assert tracker.estimate(3, rank=0).queued_token_work == 0
    assert tracker.estimate(0, rank=0).cache_hit_rate == pytest.approx(700 / 1200)


def test_throughput_counts_tokens_inside_the_window_only(loads_mod):
    clock = Clock()
    tracker = loads_mod.LoadTracker(clock=clock)
    tracker.generated(1000, rank=0)
    clock.now += 1.0
    tracker.generated(3000, rank=0)
    assert tracker.estimate(0, rank=0).gen_throughput == 2000.0
    clock.now += 1.5  # the first sample falls out of the two-second window
    assert tracker.estimate(0, rank=0).gen_throughput == 1500.0
    clock.now += 2.0
    assert tracker.estimate(0, rank=0).gen_throughput == 0.0
    tracker.generated(0, rank=0)
    assert tracker.estimate(0, rank=0).gen_throughput == 0.0


def test_the_hit_rate_averages_recent_first_outputs_and_ignores_empty_prompts(loads_mod):
    tracker = loads_mod.LoadTracker(clock=Clock())
    tracker.submitted("x", 0, rank=0)
    tracker.first_output("x", 0, 0, rank=0)
    assert tracker.estimate(0, rank=0).cache_hit_rate == 0.0
    for index in range(loads_mod.HIT_RATE_SAMPLES):
        tracker.first_output(f"old-{index}", 100, 0, rank=0)
    assert tracker.estimate(0, rank=0).cache_hit_rate == 0.0
    for index in range(loads_mod.HIT_RATE_SAMPLES):
        tracker.first_output(f"new-{index}", 100, 100, rank=0)
    assert tracker.estimate(0, rank=0).cache_hit_rate == 1.0, "the old samples aged out"
    tracker.first_output("odd", 10, 50, rank=0)
    assert tracker.estimate(0, rank=0).cache_hit_rate <= 1.0


def test_each_rank_keeps_its_own_hit_rate_throughput_and_queue(loads_mod):
    """Two ranks serve: each rank's figures are its own, and a rank that
    served nothing reads as idle, not as its peer."""
    tracker = loads_mod.LoadTracker(clock=Clock())
    tracker.submitted("a", 1000, rank=0)
    tracker.submitted("b", 400, rank=1)
    tracker.submitted("c", 600, rank=1)
    tracker.first_output("a", 1000, 0, rank=0)
    tracker.first_output("b", 400, 400, rank=1)
    tracker.generated(100, rank=0)
    tracker.generated(900, rank=1)

    rank0, rank1 = tracker.estimate(1, rank=0), tracker.estimate(1, rank=1)
    assert (rank0.cache_hit_rate, rank1.cache_hit_rate) == (0.0, 1.0)
    assert (rank0.gen_throughput, rank1.gen_throughput) == (50.0, 450.0)
    # Rank 0 has nothing waiting; rank 1's `c` waits behind `b`, with rank
    # 1's own hit rate discounting it to nothing.
    assert (rank0.queued_token_work, rank1.queued_token_work) == (0, 0)
    tracker.first_output("b", 400, 0, rank=1)
    assert tracker.estimate(1, rank=1).queued_token_work == 300
    assert tracker.estimate(1, rank=2) == loads_mod.LoadEstimate()


def test_an_unpinned_requests_queue_counts_on_the_rank_the_engine_names(loads_mod):
    """A request submitted without a rank (the engine picks one) counts
    toward no rank's queue until ``rank_of`` names where it went, then
    toward that rank, read once."""
    routed: dict[str, int] = {}
    tracker = loads_mod.LoadTracker(clock=Clock(), rank_of=routed.get)
    tracker.submitted("a", 1000, rank=None)
    tracker.submitted("b", 300, rank=None)
    tracker.submitted("c", 600, rank=1)
    assert tracker.estimate(5, rank=0).queued_token_work == 0
    assert tracker.estimate(5, rank=1).queued_token_work == 600
    routed["a"] = 0
    routed["b"] = 1
    assert tracker.estimate(5, rank=0).queued_token_work == 1000
    assert tracker.estimate(5, rank=1).queued_token_work == 900
    # Known once: a later answer does not move the request.
    routed["a"] = 1
    assert tracker.estimate(5, rank=0).queued_token_work == 1000
    tracker.first_output("a", 1000, 0, rank=0)
    assert tracker.pending() == 2
    # Without a resolver an unpinned request's queue counts nowhere.
    bare = loads_mod.LoadTracker(clock=Clock())
    bare.submitted("x", 100, rank=None)
    assert bare.estimate(1, rank=0).queued_token_work == 0


def test_the_tracker_tells_the_rank_it_learned_for_a_queued_request(loads_mod):
    """What a caller falls back on when the engine's own record of a request
    is gone by its first output (a non-streamed request's only output)."""
    routed: dict[str, int] = {}
    tracker = loads_mod.LoadTracker(clock=Clock(), rank_of=routed.get)
    tracker.submitted("a", 100, rank=None)
    tracker.submitted("p", 100, rank=1)
    assert tracker.pending_rank("a") is None
    assert tracker.pending_rank("p") == 1
    routed["a"] = 1
    assert tracker.pending_rank("a") == 1
    tracker.first_output("a", 100, 50, rank=1)
    assert tracker.pending_rank("a") is None, "no longer queued"
    assert tracker.pending_rank("unknown") is None


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


def test_the_load_record_of_another_rank_carries_its_own_counts_and_estimate(loads_mod):
    estimate = loads_mod.LoadEstimate(queued_token_work=96, gen_throughput=80.0, cache_hit_rate=0.9)
    fields = loads_mod.scheduler_load_fields(
        4, 1, 0.5, estimate, dp_rank=1, max_total_num_tokens=8_000
    )
    assert fields["dp_rank"] == 1
    assert (fields["gen_throughput"], fields["cache_hit_rate"]) == (80.0, 0.9)
    record = loads_mod.load_record(4, 1, 0.5, estimate, dp_rank=1, max_total_num_tokens=8_000)
    assert (record.running_requests, record.waiting_requests, record.token_usage) == (4, 1, 0.5)
    assert record.num_used_tokens == 4_000
    # The rank's own estimate rides its record, as the first rank's does.
    assert record.waiting_uncached_tokens == 96
    assert (record.gen_throughput, record.cache_hit_rate) == (80.0, 0.9)
