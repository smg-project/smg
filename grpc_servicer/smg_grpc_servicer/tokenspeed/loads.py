"""Conversion of TokenSpeed scheduler load replies into the gateway's protobuf.

Kept free of engine imports so the field mapping can be unit-tested without
TokenSpeed installed (see grpc_servicer/tests/test_tokenspeed_loads.py),
the same split the SGLang side uses.
"""

from __future__ import annotations

from typing import Any

from smg_grpc_proto import tokenspeed_scheduler_pb2


def running_window(server_args: Any) -> int:
    """The scheduler's configured admission window, or 0 when unknown.

    ``GetLoadReqOutput`` carries no admission bound, so the window has to come
    from the server args. TokenSpeed spells it ``max_num_seqs``; the protobuf
    field is the one SGLang fills with the same number, so a frontend reads a
    single name across both engines. It matters most on a disaggregated decode
    worker: that window is exactly the bound its prefill peer's bootstrap
    deadline is racing, and a frontend that cannot see it cannot pace its
    dispatch.
    """
    return int(
        getattr(server_args, "max_num_seqs", 0)
        or getattr(server_args, "max_running_requests", 0)
        or 0
    )


def convert_load_to_protobuf(
    load_output: Any,
    *,
    page_size: int,
    max_total_num_tokens: int,
    max_running_requests: int,
) -> tokenspeed_scheduler_pb2.SchedulerLoad:
    """Convert one rank's ``GetLoadReqOutput`` to a protobuf ``SchedulerLoad``.

    The reply counts ``num_reqs`` as running + waiting and reports KV usage in
    pages, so running requests and used tokens are derived here.
    """
    num_waiting_reqs = int(load_output.num_waiting_reqs)
    num_total_reqs = int(load_output.num_reqs)
    num_used_tokens = int(load_output.num_pages) * page_size
    return tokenspeed_scheduler_pb2.SchedulerLoad(
        dp_rank=int(load_output.dp_rank),
        num_running_reqs=max(0, num_total_reqs - num_waiting_reqs),
        num_waiting_reqs=num_waiting_reqs,
        num_total_reqs=num_total_reqs,
        num_used_tokens=num_used_tokens,
        max_total_num_tokens=max_total_num_tokens,
        token_usage=(num_used_tokens / max_total_num_tokens if max_total_num_tokens > 0 else 0.0),
        max_running_requests=max_running_requests,
    )


def convert_snapshot_to_protobuf(
    snapshot: Any,
    *,
    max_total_num_tokens: int,
    max_running_requests: int,
) -> tokenspeed_scheduler_pb2.SchedulerLoad:
    """Keep allocator occupancy separate from the model's admission capacity.

    Native page counts are LCM allocator blocks, not prefix-sized token pages.
    Hybrid packing and state reservations prevent an exact token count from
    being derived from those blocks. Express used capacity as a token-budget
    equivalent instead; admission/finish traces report actual cached tokens.
    Cached blocks are resident but reclaimable; only active blocks exert pressure.
    """
    return tokenspeed_scheduler_pb2.SchedulerLoad(
        dp_rank=int(snapshot.dp_rank),
        num_running_reqs=int(snapshot.num_running_reqs),
        num_waiting_reqs=int(snapshot.num_waiting_reqs),
        num_total_reqs=int(snapshot.num_running_reqs + snapshot.num_waiting_reqs),
        num_used_tokens=(
            int(snapshot.num_used_pages) * max_total_num_tokens // snapshot.max_total_pages
        ),
        max_total_num_tokens=max_total_num_tokens,
        token_usage=snapshot.num_used_pages / snapshot.max_total_pages,
        active_token_usage=snapshot.num_active_pages / snapshot.max_total_pages,
        max_running_requests=max_running_requests,
    )
