"""Conversion of SGLang load snapshots into the gateway's protobuf messages.

Kept free of engine imports so the field mapping can be unit-tested without
SGLang installed (see grpc_servicer/tests/test_sglang_get_loads.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_grpc_proto import sglang_scheduler_pb2

if TYPE_CHECKING:
    from sglang.srt.managers.load_snapshot import LoadSnapshot

# SGLang renamed the prefill pre-allocation queue counter to the bootstrap
# queue while the gateway proto kept the original field name. Reading only the
# old attribute raised AttributeError on every GetLoads from a disaggregated
# worker, so the gateway never received load reports from PD SGLang workers.
_PREFILL_QUEUE_ATTRS = ("prefill_bootstrap_queue_reqs", "prefill_prealloc_queue_reqs")


def _first_attr(obj: Any, names: tuple[str, ...], default: Any = 0) -> Any:
    """Return the first attribute of ``obj`` present in ``names``."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def convert_loads_to_protobuf(result: LoadSnapshot) -> sglang_scheduler_pb2.SchedulerLoad:
    """Convert a LoadSnapshot to a protobuf SchedulerLoad message."""
    scheduler_load = sglang_scheduler_pb2.SchedulerLoad(
        dp_rank=result.dp_rank,
        num_running_reqs=result.num_running_reqs,
        num_waiting_reqs=result.num_waiting_reqs,
        num_total_reqs=result.num_running_reqs + result.num_waiting_reqs,
        num_used_tokens=result.num_used_tokens,
        max_total_num_tokens=result.max_total_num_tokens,
        token_usage=result.token_usage,
        gen_throughput=result.gen_throughput,
        cache_hit_rate=result.cache_hit_rate,
        utilization=result.utilization,
        max_running_requests=result.max_running_requests,
        # Queued token-work: waiting-queue tokens not served from cache.
        num_waiting_uncached_tokens=result.num_waiting_uncached_tokens,
    )

    # Add optional sections using CopyFrom for proper protobuf assignment
    if result.memory:
        scheduler_load.memory.CopyFrom(
            sglang_scheduler_pb2.MemoryMetrics(
                weight_gb=result.memory.weight_gb,
                kv_cache_gb=result.memory.kv_cache_gb,
                graph_gb=result.memory.graph_gb,
                token_capacity=result.memory.token_capacity,
            )
        )

    if result.speculative:
        scheduler_load.speculative.CopyFrom(
            sglang_scheduler_pb2.SpeculativeMetrics(
                accept_length=result.speculative.accept_length,
                accept_rate=result.speculative.accept_rate,
            )
        )

    if result.lora:
        scheduler_load.lora.CopyFrom(
            sglang_scheduler_pb2.LoRAMetrics(
                slots_used=result.lora.slots_used,
                slots_total=result.lora.slots_total,
                utilization=result.lora.utilization,
            )
        )

    if result.disaggregation:
        disaggregation = result.disaggregation
        scheduler_load.disaggregation.CopyFrom(
            sglang_scheduler_pb2.DisaggregationMetrics(
                mode=disaggregation.mode,
                prefill_prealloc_queue_reqs=_first_attr(disaggregation, _PREFILL_QUEUE_ATTRS),
                prefill_inflight_queue_reqs=disaggregation.prefill_inflight_queue_reqs,
                decode_prealloc_queue_reqs=disaggregation.decode_prealloc_queue_reqs,
                decode_transfer_queue_reqs=disaggregation.decode_transfer_queue_reqs,
                decode_retracted_queue_reqs=disaggregation.decode_retracted_queue_reqs,
                kv_transfer_speed_gb_s=disaggregation.kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=disaggregation.kv_transfer_latency_ms,
            )
        )

    if result.queues:
        scheduler_load.queues.CopyFrom(
            sglang_scheduler_pb2.QueueMetrics(
                waiting=result.queues.waiting,
                grammar=result.queues.grammar,
                paused=result.queues.paused,
                retracted=result.queues.retracted,
            )
        )

    return scheduler_load
