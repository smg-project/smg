"""Real grpc.aio loopback TCP, production Generate/Abort and protobuf codecs.

CPU engine imports, request conversion and scheduler transport are stand-ins;
no model execution, scheduler process or GPU is involved. The scheduler stub
never returns an AbortReq, matching the running-request no-ACK contract.
"""

import asyncio
from types import SimpleNamespace

import grpc
import pytest
from smg_grpc_servicer.sglang import GRAMMAR_REJECTED_ERR_TYPE
from test_selected_token_logprobs import manager_mod, servicer  # noqa: F401
from test_sglang_request_manager_exits import _settle, manager  # noqa: F401


async def _wait_for_request(rm, request_id):
    async def wait():
        while request_id not in rm.rid_to_state:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 1)


@pytest.fixture(autouse=True)
def token_timing_stand_in(manager):  # noqa: F811
    # The shared exit fixture supplies trace stats; normal batch output also
    # records token timing. Only the engine timing boundary is substituted.
    initialize = manager._req_stats_init

    def init(*args, **kwargs):
        initialize(*args, **kwargs)
        for state in manager.rid_to_state.values():
            state.time_stats.first_token_time = 0.0
            state.time_stats.set_first_token_time = lambda: None
            state.time_stats.set_last_time = lambda: None

    manager._req_stats_init = init


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("send_fails", [False, True])
def test_running_abort_closes_real_grpc_generate(manager, servicer, stream, send_fails):  # noqa: F811
    """A local abort closes the RPC once and ignores late scheduler token rows."""
    from smg_grpc_proto import sglang_scheduler_pb2 as pb
    from smg_grpc_proto import sglang_scheduler_pb2_grpc as rpc

    async def scenario():
        servicer.request_manager = manager
        servicer._convert_generate_request = lambda request: SimpleNamespace(
            stream=request.stream, sampling_params=SimpleNamespace(n=1)
        )
        servicer._handle_epd_disaggregation_encode_request = lambda *args: None
        server = grpc.aio.server()
        rpc.add_SglangSchedulerServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        assert port
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        call = None
        try:
            await channel.channel_ready()
            client = rpc.SglangSchedulerStub(channel)
            call = client.Generate(pb.GenerateRequest(request_id="tcp", stream=stream), timeout=2)
            read = asyncio.create_task(call.read())
            await _wait_for_request(manager, "tcp")
            state = manager.rid_to_state["tcp"]
            if stream:
                # Receive one real protobuf token before aborting the open RPC.
                state.out_queue.put_nowait({"token_ids": [7], "finished": False})
                first = await read
                assert list(first.chunk.token_ids) == [7]
                read = asyncio.create_task(call.read())
            if send_fails:

                def refuse(obj):
                    raise RuntimeError("synthetic abort dispatch failure")

                manager.send_to_scheduler = SimpleNamespace(send_pyobj=refuse)
            ack = await client.Abort(pb.AbortRequest(request_id="tcp"), timeout=1)
            assert ack.success is not send_fails
            with pytest.raises(grpc.aio.AioRpcError) as exc:
                await read
            # Preserve the existing generic-abort mapping, never DEADLINE_EXCEEDED.
            assert exc.value.code() == grpc.StatusCode.INTERNAL
            assert exc.value.details() == "Request aborted"
            await _settle()
            assert state.event.is_set()
            assert not manager.rid_to_state
            # An eventual terminal scheduler batch cannot reopen the RPC.
            await manager._handle_batch_output(
                SimpleNamespace(
                    rids=["tcp"], output_ids=[[1]], prompt_tokens=[1], cached_tokens=[0]
                )
            )
        finally:
            if call is not None:
                call.cancel()
            await channel.close()
            await server.stop(0)

    asyncio.run(scenario())
    assert manager.exporter.names() == [" Req"]
    assert manager.exporter.spans[0].ended == 1


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("status", [400, 503, "grammar", None])
def test_scheduler_status_and_normal_completion_over_tcp(manager, servicer, stream, status):  # noqa: F811
    from smg_grpc_proto import sglang_scheduler_pb2 as pb
    from smg_grpc_proto import sglang_scheduler_pb2_grpc as rpc

    async def scenario():
        servicer.request_manager = manager
        servicer._convert_generate_request = lambda request: SimpleNamespace(
            stream=request.stream, sampling_params=SimpleNamespace(n=1)
        )
        servicer._handle_epd_disaggregation_encode_request = lambda *args: None
        server = grpc.aio.server()
        rpc.add_SglangSchedulerServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        call = None
        try:
            client = rpc.SglangSchedulerStub(channel)
            call = client.Generate(
                pb.GenerateRequest(request_id="result", stream=stream), timeout=2
            )
            read = asyncio.create_task(call.read())
            await _wait_for_request(manager, "result")
            if status not in (None, "grammar"):
                await manager._handle_abort_req(
                    SimpleNamespace(
                        rid="result",
                        finished_reason={
                            "type": "abort",
                            "status_code": status,
                            "message": "scheduler reason",
                        },
                    )
                )
                # Explicit abort races an already-terminal scheduler refusal.
                await manager.abort_request("result")
                with pytest.raises(grpc.aio.AioRpcError) as exc:
                    await read
                expected = (
                    grpc.StatusCode.INVALID_ARGUMENT
                    if status == 400
                    else grpc.StatusCode.UNAVAILABLE
                )
                assert exc.value.code() == expected
                assert exc.value.details() == "scheduler reason"
            else:
                await manager._handle_batch_output(
                    SimpleNamespace(
                        rids=["result"],
                        output_ids=[[7]],
                        prompt_tokens=[2],
                        completion_tokens=[1],
                        cached_tokens=[0],
                        finished_reasons=[
                            {
                                "type": "abort",
                                "message": "grammar rejected",
                                "status_code": 500,
                                "err_type": GRAMMAR_REJECTED_ERR_TYPE,
                            }
                            if status == "grammar"
                            else {"type": "length"}
                        ],
                    )
                )
                await manager.abort_request("result")
                if status == "grammar":
                    with pytest.raises(grpc.aio.AioRpcError) as exc:
                        await read
                    assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                    assert "grammar rejected" in exc.value.details()
                    assert exc.value.trailing_metadata()["smg-constraint-failed"] == "engine"
                    await _settle()
                    assert not manager.rid_to_state
                    return
                first = await read
                if stream:
                    assert list(first.chunk.token_ids) == [7]
                    complete = await call.read()
                else:
                    complete = first
                assert list(complete.complete.output_ids) == [7]
                assert complete.complete.finish_reason == "length"
                assert await call.read() is grpc.aio.EOF
                assert await call.code() == grpc.StatusCode.OK
            await _settle()
            assert not manager.rid_to_state
        finally:
            if call is not None:
                call.cancel()
            await channel.close()
            await server.stop(0)

    asyncio.run(scenario())
    assert manager.exporter.names() == [" Req"]
    assert manager.exporter.spans[0].ended == 1
