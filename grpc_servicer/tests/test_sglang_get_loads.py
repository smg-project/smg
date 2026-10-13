"""GetLoads must convert a disaggregated SGLang load snapshot.

SGLang renamed the prefill pre-allocation queue counter to
``prefill_bootstrap_queue_reqs``. The servicer read the old name, so every
GetLoads from a prefill or decode worker raised AttributeError and the gateway
never received load reports from PD SGLang workers.

Run with: pytest grpc_servicer/tests/test_sglang_get_loads.py
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")


@pytest.fixture(scope="module")
def loads_mod():
    """Load loads.py by path: the sglang package __init__ imports the engine."""
    path = Path(__file__).resolve().parent.parent / "smg_grpc_servicer" / "sglang" / "loads.py"
    spec = importlib.util.spec_from_file_location("sglang_loads_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(disaggregation):
    return SimpleNamespace(
        dp_rank=0,
        num_running_reqs=2,
        num_waiting_reqs=1,
        num_used_tokens=10,
        max_total_num_tokens=100,
        token_usage=0.1,
        gen_throughput=12.5,
        cache_hit_rate=0.25,
        utilization=0.5,
        max_running_requests=8,
        num_waiting_uncached_tokens=3,
        memory=None,
        speculative=None,
        lora=None,
        disaggregation=disaggregation,
        queues=None,
    )


def _disaggregation(**prefill_queue):
    return SimpleNamespace(
        mode="prefill",
        prefill_inflight_queue_reqs=4,
        decode_prealloc_queue_reqs=0,
        decode_transfer_queue_reqs=0,
        decode_retracted_queue_reqs=0,
        kv_transfer_speed_gb_s=1.5,
        kv_transfer_latency_ms=2.5,
        **prefill_queue,
    )


class TestDisaggregationMetrics:
    def test_bootstrap_queue_maps_to_prealloc_field(self, loads_mod):
        load = loads_mod.convert_loads_to_protobuf(
            _snapshot(_disaggregation(prefill_bootstrap_queue_reqs=3))
        )

        assert load.disaggregation.mode == "prefill"
        assert load.disaggregation.prefill_prealloc_queue_reqs == 3
        assert load.disaggregation.prefill_inflight_queue_reqs == 4
        assert load.num_total_reqs == 3

    def test_legacy_prealloc_queue_still_converts(self, loads_mod):
        load = loads_mod.convert_loads_to_protobuf(
            _snapshot(_disaggregation(prefill_prealloc_queue_reqs=7))
        )

        assert load.disaggregation.prefill_prealloc_queue_reqs == 7

    def test_regular_worker_has_no_disaggregation_section(self, loads_mod):
        load = loads_mod.convert_loads_to_protobuf(_snapshot(None))

        assert not load.HasField("disaggregation")


class TestEffectiveRunningWindow:
    """The window the schedulers enforce, off their published snapshots (the
    figure `GetServerInfo` reports when `--max-running-requests` is unset)."""

    def test_the_smallest_published_window_is_the_bound(self, loads_mod):
        first, second = _snapshot(None), _snapshot(None)
        first.max_running_requests, second.max_running_requests = 1574, 314
        second.dp_rank = 1
        assert loads_mod.effective_running_window([first, second]) == 314

    def test_the_flags_units_are_the_share_times_the_attention_dp_width(self, loads_mod):
        # The engine splits --max-running-requests across its attention-DP
        # workers and each publishes its share: 16384 over 8 publishes 2048 each.
        ranks = [_snapshot(None) for _ in range(8)]
        for rank, snapshot in enumerate(ranks):
            snapshot.dp_rank, snapshot.max_running_requests = rank, 2048
        assert loads_mod.effective_running_window(ranks, attention_dp_size=8) == 16384
        # Without attention DP every scheduler runs the whole flag.
        assert loads_mod.effective_running_window(ranks, attention_dp_size=1) == 2048
        assert loads_mod.effective_running_window([], attention_dp_size=8) == 0

    def test_the_figure_is_clamped_to_what_the_gateway_label_can_carry(self, loads_mod):
        # The gateway parses the label as a 16-bit count: 32 attention-DP
        # workers publishing 2048 each (65536) would read as no window.
        ranks = [_snapshot(None) for _ in range(32)]
        for rank, snapshot in enumerate(ranks):
            snapshot.dp_rank, snapshot.max_running_requests = rank, 2048
        assert loads_mod.effective_running_window(ranks, attention_dp_size=32) == 65535
        assert loads_mod.effective_running_window(ranks, attention_dp_size=31) == 63488
        wide = _snapshot(None)
        wide.max_running_requests = 70000
        assert loads_mod.effective_running_window([wide]) == 65535
        assert loads_mod.MAX_REPORTED_RUNNING_WINDOW == 65535

    def test_the_ranks_that_published_a_window_are_counted(self, loads_mod):
        published, unset = _snapshot(None), _snapshot(None)
        unset.max_running_requests = 0
        assert loads_mod.ranks_with_a_window([]) == 0
        assert loads_mod.ranks_with_a_window([published, unset]) == 1
        assert loads_mod.ranks_with_a_window([published, published]) == 2

    def test_no_snapshot_or_no_figure_means_no_window(self, loads_mod):
        unset = _snapshot(None)
        unset.max_running_requests = 0
        assert loads_mod.effective_running_window([]) == 0
        assert loads_mod.effective_running_window([unset]) == 0
        # A rank that has not computed one yet does not hide its peers' bound.
        published = _snapshot(None)
        assert loads_mod.effective_running_window([unset, published]) == 8


class TestLoadRecord:
    """The record the KV-event stream carries: the published snapshot of the
    batch's rank, converted as GetLoads converts it."""

    def _snapshots(self):
        first, second = _snapshot(None), _snapshot(_disaggregation(prefill_bootstrap_queue_reqs=4))
        second.dp_rank = 1
        second.num_running_reqs = 9
        return [first, second]

    def test_the_batch_rank_selects_its_snapshot(self, loads_mod):
        record = loads_mod.load_record(self._snapshots(), 1)
        assert record.running_requests == 9
        assert record.disaggregation.prefill_prealloc_queue_reqs == 4
        assert record.waiting_uncached_tokens == 3  # the scheduler's queued token-work
        assert record.cache_hit_rate == 0.25 and record.max_total_num_tokens == 100

    def test_a_batch_without_a_rank_reads_the_first_rank(self, loads_mod):
        snapshots = list(reversed(self._snapshots()))
        assert loads_mod.load_record(snapshots, None).running_requests == 2

    def test_a_batch_naming_an_unpublished_rank_gets_no_record(self, loads_mod):
        # Rank 7 has published nothing: no record, not the first rank's figures.
        assert loads_mod.load_record(self._snapshots(), 7) is None

    def test_no_published_snapshot_means_no_record(self, loads_mod):
        assert loads_mod.load_record([], 0) is None
