"""Engine-free coverage of vLLM cache flush and the generated gRPC contract."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from smg_grpc_proto.generated import common_pb2

_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "admin.py"
_spec = importlib.util.spec_from_file_location("vllm_admin", _PATH)
admin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(admin)


def context(remaining=None):
    return Mock(time_remaining=Mock(return_value=remaining))


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_immediate_reset_preserves_engine_result(success):
    engine = SimpleNamespace(reset_prefix_cache=AsyncMock(return_value=success))
    ctx = context()
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(), ctx)
    assert result.success is success
    engine.reset_prefix_cache.assert_awaited_once_with(
        reset_running_requests=False, reset_connector=False
    )
    ctx.set_code.assert_not_called()


@pytest.mark.asyncio
async def test_wait_retries_until_engine_accepts(monkeypatch):
    monkeypatch.setattr(admin, "_RETRY_INTERVAL_S", 0)
    engine = SimpleNamespace(reset_prefix_cache=AsyncMock(side_effect=[False, False, True]))
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(timeout_s=1), context())
    assert result.success
    assert engine.reset_prefix_cache.await_count == 3


@pytest.mark.asyncio
async def test_busy_wait_expires():
    engine = SimpleNamespace(reset_prefix_cache=AsyncMock(return_value=False))
    ctx = context()
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(timeout_s=0.01), ctx)
    assert not result.success
    ctx.set_code.assert_called_once_with(grpc.StatusCode.DEADLINE_EXCEEDED)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_s,remaining", [(0, None), (0.01, None), (10, 0.01)])
async def test_unresponsive_engine_is_bounded(monkeypatch, timeout_s, remaining):
    monkeypatch.setattr(admin, "_RESET_RPC_TIMEOUT_S", 0.01)
    cancelled = asyncio.Event()

    async def reset(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    ctx = context(remaining)
    result = await asyncio.wait_for(
        admin.flush_cache(
            SimpleNamespace(reset_prefix_cache=reset),
            common_pb2.FlushCacheRequest(timeout_s=timeout_s),
            ctx,
        ),
        timeout=1,
    )
    assert not result.success
    assert cancelled.is_set()
    ctx.set_code.assert_called_once_with(grpc.StatusCode.DEADLINE_EXCEEDED)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_s", [-1, float("nan"), float("inf")])
async def test_invalid_timeout_never_calls_engine(timeout_s):
    engine = SimpleNamespace(reset_prefix_cache=AsyncMock())
    ctx = context()
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(timeout_s=timeout_s), ctx)
    assert not result.success
    engine.reset_prefix_cache.assert_not_called()
    ctx.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)


@pytest.mark.asyncio
async def test_cancellation_propagates():
    engine = SimpleNamespace(reset_prefix_cache=AsyncMock(side_effect=asyncio.CancelledError))
    ctx = context()
    with pytest.raises(asyncio.CancelledError):
        await admin.flush_cache(engine, common_pb2.FlushCacheRequest(), ctx)
    ctx.set_code.assert_not_called()


@pytest.mark.asyncio
async def test_generated_rpc_round_trip():
    engine = SimpleNamespace(reset_prefix_cache=AsyncMock(side_effect=[True, False]))

    class Servicer(vllm_engine_pb2_grpc.VllmEngineServicer):
        async def FlushCache(self, request, context):
            return await admin.flush_cache(engine, request, context)

    method = vllm_engine_pb2.DESCRIPTOR.services_by_name["VllmEngine"].methods_by_name["FlushCache"]
    assert method.input_type.full_name == "smg.grpc.common.FlushCacheRequest"
    server = grpc.aio.server()
    vllm_engine_pb2_grpc.add_VllmEngineServicer_to_server(Servicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)
            assert (await stub.FlushCache(common_pb2.FlushCacheRequest(), timeout=1)).success
            assert not (await stub.FlushCache(common_pb2.FlushCacheRequest(), timeout=1)).success
    finally:
        await server.stop(None)


def dp_engine(outcomes):
    # Model vLLM's lossy public API: it reports only the first rank's result.
    ranks = {
        rank.to_bytes(2, "little"): AsyncMock(side_effect=values)
        for rank, values in outcomes.items()
    }

    async def call(method, reset_running_requests, reset_connector, *, engine):
        assert (method, reset_running_requests, reset_connector) == (
            "reset_prefix_cache",
            False,
            False,
        )
        return await ranks[engine]()

    core = SimpleNamespace(core_engines=list(ranks), _call_utility_async=call)

    async def public_reset(**kwargs):
        results = await asyncio.gather(
            *(call("reset_prefix_cache", False, False, engine=rank) for rank in ranks)
        )
        return results[0]

    return SimpleNamespace(
        engine_core=core, reset_prefix_cache=AsyncMock(side_effect=public_reset)
    ), ranks


@pytest.mark.asyncio
@pytest.mark.parametrize("outcomes", [{0: [True], 1: [False]}, {4: [False], 7: [True]}])
async def test_dp_partial_reset_is_not_success(outcomes):
    engine, ranks = dp_engine(outcomes)
    ctx = context()
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(), ctx)
    assert not result.success
    assert all(rank.await_count == 1 for rank in ranks.values())
    engine.reset_prefix_cache.assert_not_called()
    ctx.set_code.assert_not_called()


@pytest.mark.asyncio
async def test_dp_retry_requires_all_ranks_to_succeed_in_same_attempt(monkeypatch):
    monkeypatch.setattr(admin, "_RETRY_INTERVAL_S", 0)
    engine, ranks = dp_engine({0: [True, False, True], 1: [False, True, True]})
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(timeout_s=1), context())
    assert result.success
    assert all(rank.await_count == 3 for rank in ranks.values())


@pytest.mark.asyncio
async def test_dp_nonfirst_rank_error_is_reported():
    engine, ranks = dp_engine({0: [True], 1: [RuntimeError("rank failed")]})
    ctx = context()
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(), ctx)
    assert not result.success
    assert all(rank.await_count == 1 for rank in ranks.values())
    ctx.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)


@pytest.mark.asyncio
async def test_dp_hung_rank_times_out_and_cancels_pending_call():
    engine, ranks = dp_engine({0: [True], 1: [True]})
    cancelled = asyncio.Event()

    async def hang():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    ranks[b"\x01\x00"].side_effect = hang
    ctx = context()
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(timeout_s=0.01), ctx)
    assert not result.success
    assert cancelled.is_set()
    ctx.set_code.assert_called_once_with(grpc.StatusCode.DEADLINE_EXCEEDED)


@pytest.mark.asyncio
async def test_external_dp_single_managed_rank_uses_public_api():
    engine, ranks = dp_engine({7: [True]})
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(), context())
    assert result.success
    engine.reset_prefix_cache.assert_awaited_once_with(
        reset_running_requests=False, reset_connector=False
    )
    assert next(iter(ranks.values())).await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("identities", [[], [b"\x00\x00", b"\x01\x00"]])
async def test_dp_missing_rank_interface_does_not_fall_back_to_lossy_api(identities):
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(core_engines=identities),
        reset_prefix_cache=AsyncMock(return_value=True),
    )
    ctx = context()
    result = await admin.flush_cache(engine, common_pb2.FlushCacheRequest(), ctx)
    assert not result.success
    engine.reset_prefix_cache.assert_not_called()
    ctx.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
