"""Engine-free coverage of the vLLM profiler RPCs and the generated gRPC contract."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
from smg_grpc_proto import vllm_engine_pb2_grpc
from smg_grpc_proto.generated import common_pb2

_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "admin.py"
_spec = importlib.util.spec_from_file_location("vllm_admin_profile", _PATH)
admin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(admin)


def context(remaining=None):
    return Mock(time_remaining=Mock(return_value=remaining), abort=AsyncMock())


def engine():
    return SimpleNamespace(start_profile=AsyncMock(), stop_profile=AsyncMock())


def test_the_generated_contract_carries_both_rpcs():
    assert hasattr(vllm_engine_pb2_grpc.VllmEngineServicer, "StartProfile")
    assert hasattr(vllm_engine_pb2_grpc.VllmEngineServicer, "StopProfile")


@pytest.mark.asyncio
async def test_a_start_and_a_stop_drive_the_engines_profiler():
    eng, ctx = engine(), context()
    started = await admin.start_profile(eng, common_pb2.StartProfileRequest(), ctx)
    assert started.success and started.message == "Start profiling succeeded"
    eng.start_profile.assert_awaited_once_with()
    stopped = await admin.stop_profile(eng, ctx)
    assert stopped.success and stopped.message == "Stop profiling succeeded"
    eng.stop_profile.assert_awaited_once_with()
    ctx.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_launch_time_options_are_refused_before_any_call():
    eng, ctx = engine(), context()
    request = common_pb2.StartProfileRequest(output_dir="/traces", num_steps=3)
    refused = await admin.start_profile(eng, request, ctx)
    assert not refused.success
    ctx.abort.assert_awaited_once()
    code, message = ctx.abort.await_args.args
    assert code is grpc.StatusCode.INVALID_ARGUMENT
    assert "output_dir, num_steps" in message
    eng.start_profile.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_engines_failure_is_internal_with_its_words():
    eng, ctx = engine(), context()
    eng.start_profile = AsyncMock(side_effect=RuntimeError("Profiler is not enabled."))
    failed = await admin.start_profile(eng, common_pb2.StartProfileRequest(), ctx)
    assert not failed.success
    code, message = ctx.abort.await_args.args
    assert code is grpc.StatusCode.INTERNAL
    assert "Profiler is not enabled." in message


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [None, 0.01], ids=["own bound", "the client's deadline"])
async def test_the_wait_is_bounded(monkeypatch, remaining):
    monkeypatch.setattr(admin, "PROFILE_TIMEOUT_S", 0.01 if remaining is None else 10.0)

    async def never():
        await asyncio.Event().wait()

    eng = SimpleNamespace(start_profile=never, stop_profile=never)
    ctx = context(remaining)
    timed_out = await admin.stop_profile(eng, ctx)
    assert not timed_out.success
    code, message = ctx.abort.await_args.args
    assert code is grpc.StatusCode.DEADLINE_EXCEEDED
    assert message.startswith("Stop profiling timed out after 0.01s")
