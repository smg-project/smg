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
