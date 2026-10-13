"""The facts the SGLang scheduler plugin reports at the engine handshake of the
Rust servicer path, from the scheduler's resolved arguments: the running window
in the flag's units, the scheduler count and one attention group's TP width, on
either engine release."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# The wire module rides on pyzmq and msgspec; the plain unit-test job installs
# msgspec but not pyzmq and skips this module, as it does the wire's own tests.
pytest.importorskip("msgspec")
pytest.importorskip("zmq")
from smg_grpc_servicer.sglang import zmq_msgpack as wire  # noqa: E402


def _scheduler(**server_args):
    args = SimpleNamespace(
        tp_size=16, pp_size=1, dp_size=1, enable_dp_attention=False, served_model_name="served"
    )
    for key, value in server_args.items():
        setattr(args, key, value)
    return SimpleNamespace(
        server_args=args,
        page_size=16,
        max_total_num_tokens=4096,
        chunked_prefill_size=2048,
        max_running_requests=2048,
        ps=SimpleNamespace(dp_rank=0),
        model_config=SimpleNamespace(context_len=8192, dtype="bfloat16", model_path="org/model"),
    )


def _facts(ready):
    return (
        ready.data_parallel_size,
        ready.tensor_parallel_size,
        ready.max_num_seqs,
        ready.world_size,
    )


def test_the_handshake_reports_the_window_in_the_flags_units_on_either_release():
    """Eight attention-DP workers computing 2048 each: the window goes out as
    16384 (the flag's units, what an explicit flag and GetServerInfo report),
    eight schedulers dial, and each attention group runs a TP width of 2 out
    of the group of 16, whether the engine keeps the flag as given (the pinned
    release) or resolves it into attn_dp_size (the newer releases)."""
    pinned = wire.ready_response_for_scheduler(_scheduler(dp_size=8, enable_dp_attention=True))
    assert _facts(pinned) == (8, 2, 16384, 16)
    rewritten = wire.ready_response_for_scheduler(
        _scheduler(attn_dp_size=8, dp_size=1, enable_dp_attention=False)
    )
    assert _facts(rewritten) == (8, 2, 16384, 16)
    # Attention context parallelism (the newer releases) splits one attention
    # group's TP width further and leaves the window and the scheduler count alone.
    context = wire.ready_response_for_scheduler(
        _scheduler(attn_dp_size=4, attn_cp_size=2, dp_size=1, enable_dp_attention=False)
    )
    assert _facts(context) == (4, 2, 8192, 16)


def test_plain_data_parallelism_and_a_single_scheduler_report_their_own_figures():
    """Replicas each run the whole flag and the whole TP group; a single
    scheduler's bound is the flag."""
    replicas = wire.ready_response_for_scheduler(_scheduler(dp_size=8))
    assert _facts(replicas) == (8, 16, 2048, 16)
    single = wire.ready_response_for_scheduler(_scheduler())
    assert _facts(single) == (1, 16, 2048, 16)
    assert (single.max_model_len, single.block_size, single.num_gpu_blocks) == (8192, 16, 256)
    assert single.instance_id == "served" and single.data_parallel_rank == 0


def test_each_scheduler_reports_its_own_rank_on_either_release():
    """The pinned release's dp_rank is the attention-DP rank. The newer
    releases keep the attention group's rank in attn_dp_rank: it is the
    scheduler's rank for one replica, whichever index dp_rank carries (the engine runs one of
    dp_size and attn_dp_size at 1), so two schedulers of one engine never share an identity or a
    data_parallel_rank."""
    pinned = _scheduler(dp_size=8, enable_dp_attention=True)
    pinned.ps = SimpleNamespace(dp_rank=3)
    assert wire.scheduler_dp_rank(pinned) == 3
    first = _scheduler(attn_dp_size=8, dp_size=1, enable_dp_attention=False)
    first.ps = SimpleNamespace(dp_rank=0, attn_dp_rank=0)
    second = _scheduler(attn_dp_size=8, dp_size=1, enable_dp_attention=False)
    second.ps = SimpleNamespace(dp_rank=0, attn_dp_rank=1)
    assert (wire.scheduler_dp_rank(first), wire.scheduler_dp_rank(second)) == (0, 1)
    assert wire.ready_response_for_scheduler(second).data_parallel_rank == 1
    # A release that numbers dp_rank as the flat index too: attn_dp_rank is the rank.
    flat = _scheduler(attn_dp_size=8, dp_size=1, enable_dp_attention=False)
    flat.ps = SimpleNamespace(dp_rank=5, attn_dp_rank=5)
    assert wire.scheduler_dp_rank(flat) == 5
    # Plain data parallelism on those releases resolves to attn_dp_size=1 and
    # keeps dp_rank, the replica index (attn_dp_rank is 0 in every replica).
    plain = _scheduler(dp_size=8, attn_dp_size=1, enable_dp_attention=False)
    plain.ps = SimpleNamespace(dp_rank=5, attn_dp_rank=0)
    assert wire.scheduler_dp_rank(plain) == 5
    assert wire.ready_response_for_scheduler(plain).data_parallel_rank == 5
    # A state that carries attn_dp_rank on a release that did not resolve
    # attn_dp_size keeps dp_rank as the rank.
    unresolved = _scheduler(dp_size=8, enable_dp_attention=True)
    unresolved.ps = SimpleNamespace(dp_rank=5, attn_dp_rank=5)
    assert wire.scheduler_dp_rank(unresolved) == 5
