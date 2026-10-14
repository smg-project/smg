"""The load record the KV-event stream carries (``smg_grpc_servicer.load_record``):
the conversion from each engine's ``GetLoads`` message and the recorder that
numbers and paces one stream's records as the Rust servicer's relay does. No
engine needed."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("smg_grpc_proto")

from smg_grpc_proto import (  # noqa: E402
    sglang_scheduler_pb2,
    tokenspeed_scheduler_pb2,
    vllm_engine_pb2,
)
from smg_grpc_proto.generated import common_pb2  # noqa: E402
from smg_grpc_servicer import load_record  # noqa: E402


def _record(**fields) -> common_pb2.EngineLoad:
    return common_pb2.EngineLoad(**fields)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


# ---------------------------------------------------------------------------
# engine_load_from: the record from each engine's GetLoads message
# ---------------------------------------------------------------------------


def test_the_vllm_report_gives_the_core_the_hit_rate_and_the_token_counts():
    load = vllm_engine_pb2.SchedulerLoad(
        num_running_reqs=3,
        num_waiting_reqs=2,
        num_total_reqs=5,
        num_used_tokens=4_096,
        max_total_num_tokens=65_536,
        token_usage=0.0625,
        gen_throughput=410.5,
        cache_hit_rate=0.3,
        utilization=0.0625,
        max_running_requests=256,
        num_waiting_uncached_tokens=1_500,
    )
    record = load_record.engine_load_from(load, queued_token_work=True)
    assert (record.running_requests, record.waiting_requests) == (3, 2)
    assert (record.token_usage, record.gen_throughput) == (0.0625, 410.5)
    assert record.max_running_requests == 256
    assert record.waiting_uncached_tokens == 1_500
    assert (record.cache_hit_rate, record.num_used_tokens, record.max_total_num_tokens) == (
        0.3,
        4_096,
        65_536,
    )
    assert not record.HasField("active_token_usage")
    for section in ("memory", "queues", "speculative", "lora", "disaggregation"):
        assert not record.HasField(section)


def test_the_queued_token_work_is_absent_not_zero_when_the_servicer_cannot_tell():
    load = tokenspeed_scheduler_pb2.SchedulerLoad(num_running_reqs=1)
    record = load_record.engine_load_from(load, queued_token_work=False)
    assert not record.HasField("waiting_uncached_tokens")
    told = load_record.engine_load_from(load, queued_token_work=True)
    assert told.HasField("waiting_uncached_tokens") and told.waiting_uncached_tokens == 0


def test_the_sglang_report_carries_every_section_it_has():
    load = sglang_scheduler_pb2.SchedulerLoad(
        dp_rank=2,
        num_running_reqs=7,
        num_waiting_reqs=3,
        num_used_tokens=12_000,
        max_total_num_tokens=64_000,
        token_usage=0.1875,
        gen_throughput=1_500.5,
        cache_hit_rate=0.42,
        max_running_requests=128,
        num_waiting_uncached_tokens=900,
        memory=sglang_scheduler_pb2.MemoryMetrics(
            weight_gb=14.0, kv_cache_gb=20.5, graph_gb=1.5, token_capacity=64_000
        ),
        queues=sglang_scheduler_pb2.QueueMetrics(waiting=3, grammar=1, paused=0, retracted=2),
        speculative=sglang_scheduler_pb2.SpeculativeMetrics(accept_length=2.5, accept_rate=0.8),
        lora=sglang_scheduler_pb2.LoRAMetrics(slots_used=1, slots_total=4, utilization=0.25),
        disaggregation=sglang_scheduler_pb2.DisaggregationMetrics(
            mode="prefill",
            prefill_prealloc_queue_reqs=1,
            prefill_inflight_queue_reqs=2,
            decode_prealloc_queue_reqs=3,
            decode_transfer_queue_reqs=4,
            decode_retracted_queue_reqs=5,
            kv_transfer_speed_gb_s=6.5,
            kv_transfer_latency_ms=7.5,
        ),
    )
    record = load_record.engine_load_from(load, queued_token_work=True)
    assert record.waiting_uncached_tokens == 900
    assert record.memory == common_pb2.EngineMemory(
        weight_gb=14.0, kv_cache_gb=20.5, graph_gb=1.5, token_capacity=64_000
    )
    assert record.queues == common_pb2.EngineQueues(waiting=3, grammar=1, paused=0, retracted=2)
    assert record.speculative == common_pb2.EngineSpeculative(accept_length=2.5, accept_rate=0.8)
    assert record.lora == common_pb2.EngineLora(slots_used=1, slots_total=4, utilization=0.25)
    assert record.disaggregation == common_pb2.EngineDisaggregation(
        mode="prefill",
        prefill_prealloc_queue_reqs=1,
        prefill_inflight_queue_reqs=2,
        decode_prealloc_queue_reqs=3,
        decode_transfer_queue_reqs=4,
        decode_retracted_queue_reqs=5,
        kv_transfer_speed_gb_s=6.5,
        kv_transfer_latency_ms=7.5,
    )


def test_the_tokenspeed_report_keeps_the_active_usage_and_its_sections():
    load = tokenspeed_scheduler_pb2.SchedulerLoad(
        num_running_reqs=1,
        num_waiting_reqs=0,
        token_usage=0.5,
        active_token_usage=0.25,
        max_running_requests=16,
        memory=tokenspeed_scheduler_pb2.MemoryMetrics(kv_cache_gb=8.0),
        queues=tokenspeed_scheduler_pb2.QueueMetrics(waiting=0),
    )
    record = load_record.engine_load_from(load, queued_token_work=False)
    assert record.HasField("active_token_usage") and record.active_token_usage == 0.25
    assert record.memory.kv_cache_gb == 8.0 and record.HasField("queues")
    assert not record.HasField("speculative")


def test_a_negative_count_reads_as_zero():
    load = vllm_engine_pb2.SchedulerLoad(num_running_reqs=-1, max_running_requests=-5)
    record = load_record.engine_load_from(load, queued_token_work=True)
    assert (record.running_requests, record.max_running_requests) == (0, 0)


def test_core_only_clears_the_telemetry_and_keeps_the_core():
    record = _record(
        running_requests=2,
        waiting_uncached_tokens=10,
        token_usage=0.5,
        active_token_usage=0.4,
        cache_hit_rate=0.9,
        num_used_tokens=5,
        max_total_num_tokens=10,
        memory=common_pb2.EngineMemory(kv_cache_gb=1.0),
        queues=common_pb2.EngineQueues(waiting=1),
    )
    load_record.core_only(record)
    assert record == _record(
        running_requests=2, waiting_uncached_tokens=10, token_usage=0.5, active_token_usage=0.4
    )


# ---------------------------------------------------------------------------
# load_changed: what is worth a load_only batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("last", "current", "changed"),
    [
        (_record(running_requests=1), _record(running_requests=2), True),
        (_record(waiting_requests=0), _record(waiting_requests=1), True),
        (_record(max_running_requests=8), _record(max_running_requests=9), True),
        (_record(), _record(waiting_uncached_tokens=0), True),
        (_record(token_usage=0.500), _record(token_usage=0.504), False),
        (_record(token_usage=0.500), _record(token_usage=0.506), True),
        (_record(token_usage=0.5), _record(token_usage=0.5, active_token_usage=0.497), False),
        (_record(token_usage=0.5), _record(token_usage=0.5, active_token_usage=0.45), True),
        (_record(gen_throughput=1000.0), _record(gen_throughput=1040.0), False),
        (_record(gen_throughput=1000.0), _record(gen_throughput=1060.0), True),
        (_record(gen_throughput=100.0), _record(gen_throughput=106.0), True),
        (_record(gen_throughput=100.0), _record(gen_throughput=104.0), False),
    ],
)
def test_load_changed_thresholds(last, current, changed):
    assert load_record.load_changed(last, current) is changed


# ---------------------------------------------------------------------------
# LoadRecorder: numbering, telemetry trimming and the heartbeat cadence
# ---------------------------------------------------------------------------


class Source:
    """A load source the tests move: the record for any rank, counted."""

    def __init__(self):
        self.record = _record(
            running_requests=1,
            token_usage=0.25,
            cache_hit_rate=0.5,
            num_used_tokens=100,
            max_total_num_tokens=400,
        )
        self.asked: list[int | None] = []

    def __call__(self, rank):
        self.asked.append(rank)
        if self.record is None:
            return None
        copy = common_pb2.EngineLoad()
        copy.CopyFrom(self.record)
        return copy


def _batch(seq: int, rank: int | None = None) -> common_pb2.KvEventBatch:
    batch = common_pb2.KvEventBatch(sequence_number=seq, timestamp=1.0)
    if rank is not None:
        batch.dp_rank = rank
    return batch


def _recorder(source, clock):
    return load_record.LoadRecorder(
        source, load_tick=0.1, heartbeat_interval=1.0, heartbeat_backoff=5.0, clock=clock
    )


def test_every_batch_carries_the_record_for_its_rank_numbered_in_order():
    source, clock = Source(), Clock()
    recorder = _recorder(source, clock)
    first = recorder.stamp(_batch(1, rank=1))
    second = recorder.stamp(_batch(2, rank=0))
    assert source.asked == [1, 0]
    assert (first.load.sample, second.load.sample) == (1, 2)
    assert not first.load.load_only and not second.load.load_only
    assert first.load.running_requests == 1 and second.load.running_requests == 1


def test_the_telemetry_rides_the_first_record_and_the_heartbeats_only():
    source, clock = Source(), Clock()
    recorder = _recorder(source, clock)
    first = recorder.stamp(_batch(1))
    assert first.load.cache_hit_rate == 0.5 and first.load.max_total_num_tokens == 400
    second = recorder.stamp(_batch(2))
    assert not second.load.HasField("cache_hit_rate")
    assert not second.load.HasField("max_total_num_tokens")
    assert second.load.token_usage == 0.25  # the core stays
    clock.now += 1.0
    beat = recorder.heartbeat()
    assert beat is not None and beat.load.load_only
    assert beat.load.cache_hit_rate == 0.5 and beat.load.num_used_tokens == 100


def test_no_heartbeat_before_the_first_batch_went_out():
    source, clock = Source(), Clock()
    recorder = _recorder(source, clock)
    clock.now += 60.0
    assert recorder.heartbeat() is None
    assert source.asked == []


def test_a_heartbeat_repeats_the_last_sequence_and_rank_after_the_interval():
    source, clock = Source(), Clock()
    recorder = _recorder(source, clock)
    recorder.stamp(_batch(7, rank=3))
    clock.now += 0.9
    assert recorder.heartbeat() is None
    clock.now += 0.1
    beat = recorder.heartbeat()
    assert beat is not None
    assert (beat.sequence_number, beat.dp_rank, beat.events) == (7, 3, [])
    assert (beat.load.sample, beat.load.load_only) == (2, True)
    assert source.asked[-1] == 3


def test_the_heartbeat_backs_off_after_two_unchanged_records_and_returns_on_a_change():
    source, clock = Source(), Clock()
    recorder = _recorder(source, clock)
    recorder.stamp(_batch(1))
    beats = []
    for _ in range(12):
        clock.now += 1.0
        beat = recorder.heartbeat()
        if beat is not None:
            beats.append(clock.now)
    # Two heartbeats at the interval, then one at the backoff.
    assert beats == [101.0, 102.0, 107.0, 112.0]
    source.record.running_requests = 2
    clock.now += 0.1
    beat = recorder.heartbeat()
    assert beat is not None and beat.load.running_requests == 2
    clock.now += 1.0
    assert recorder.heartbeat() is not None  # the cadence is back at the interval


def test_a_changed_record_goes_out_at_once_as_a_load_only_batch():
    source, clock = Source(), Clock()
    recorder = _recorder(source, clock)
    recorder.stamp(_batch(1))
    clock.now += 0.1
    assert recorder.heartbeat() is None
    source.record.waiting_requests = 4
    clock.now += 0.1
    beat = recorder.heartbeat()
    assert beat is not None and beat.load.waiting_requests == 4 and beat.load.load_only
    assert beat.sequence_number == 1


def test_a_source_without_a_record_leaves_the_batch_bare_and_sends_no_heartbeat():
    source, clock = Source(), Clock()
    source.record = None
    recorder = _recorder(source, clock)
    batch = recorder.stamp(_batch(1))
    assert not batch.HasField("load")
    clock.now += 5.0
    assert recorder.heartbeat() is None


def test_the_poll_timeout_is_the_tick_or_the_shorter_receive_timeout():
    recorder = _recorder(Source(), Clock())
    assert recorder.poll_timeout(1.0) == pytest.approx(0.1)
    assert recorder.poll_timeout(0.05) == pytest.approx(0.05)


def _messages(caplog, level):
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def test_a_source_that_cannot_be_read_leaves_the_batch_without_a_record(caplog):
    """An outage of the source: the batches go out without a record and no
    heartbeat is due, as before; the first failed read is a warning with the
    traceback, the repeats debug lines with their count, the read that ends
    the outage an info line with the count."""
    source, clock = Source(), Clock()
    failing = False

    def read(rank):
        if failing:
            raise RuntimeError("snapshot gone")
        return source(rank)

    recorder = _recorder(read, clock)
    with caplog.at_level(logging.DEBUG, logger=load_record.logger.name):
        assert recorder.stamp(_batch(1)).load.sample == 1
        failing = True
        assert not recorder.stamp(_batch(2)).HasField("load")
        clock.now += 10.0
        assert recorder.heartbeat() is None
        assert not recorder.stamp(_batch(3)).HasField("load")
        failing = False
        assert recorder.stamp(_batch(4)).load.sample == 2  # the numbering continues
    assert _messages(caplog, logging.WARNING) == [
        "load record unavailable for the stream (no rank); the KV-event stream carries cache "
        "events only until it can be read again"
    ]
    assert "RuntimeError: snapshot gone" in caplog.text
    assert _messages(caplog, logging.DEBUG) == [
        "load record unavailable for the stream (no rank) (2 in a row)",
        "load record unavailable for the stream (no rank) (3 in a row)",
    ]
    assert _messages(caplog, logging.INFO) == [
        "load record for the stream (no rank) available again after 3 failed read(s)"
    ]
    # A second outage is reported again.
    failing = True
    with caplog.at_level(logging.DEBUG, logger=load_record.logger.name):
        assert not recorder.stamp(_batch(5)).HasField("load")
    assert len(_messages(caplog, logging.WARNING)) == 2


def test_one_ranks_outage_is_reported_once_while_another_rank_reads_fine(caplog):
    """One recorder serves every rank of a relayed stream: a rank whose read
    keeps failing while another reads fine is one outage (one warning, one
    info line when it ends), not one per interleaved batch."""
    source, clock = Source(), Clock()
    broken = {0}

    def read(rank):
        if rank in broken:
            raise RuntimeError(f"rank {rank} snapshot gone")
        return source(rank)

    recorder = _recorder(read, clock)
    with caplog.at_level(logging.DEBUG, logger=load_record.logger.name):
        for seq in range(1, 7):  # ranks 1, 0, 1, 0, 1, 0
            assert recorder.stamp(_batch(seq, rank=seq % 2)).HasField("load") == (seq % 2 == 1)
        broken.clear()
        assert recorder.stamp(_batch(7, rank=0)).HasField("load")
    assert _messages(caplog, logging.WARNING) == [
        "load record unavailable for rank 0; the KV-event stream carries cache events only "
        "until it can be read again"
    ]
    assert _messages(caplog, logging.DEBUG) == [
        "load record unavailable for rank 0 (2 in a row)",
        "load record unavailable for rank 0 (3 in a row)",
    ]
    assert _messages(caplog, logging.INFO) == [
        "load record for rank 0 available again after 3 failed read(s)"
    ]


def test_a_source_with_nothing_to_report_yet_is_not_an_outage(caplog):
    """``None`` from the source (the engine is not up yet) is the known quiet
    case: no record, no heartbeat, nothing logged."""
    recorder = _recorder(lambda rank: None, Clock())
    with caplog.at_level(logging.DEBUG, logger=load_record.logger.name):
        assert not recorder.stamp(_batch(1)).HasField("load")
        assert recorder.heartbeat() is None
    assert caplog.records == []
