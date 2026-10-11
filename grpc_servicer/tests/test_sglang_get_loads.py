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
