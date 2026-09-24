"""Exercise the SGLang bridge over real ZMQ and gRPC without loading an engine."""

import ast
import asyncio
import importlib.util
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

pytest.importorskip("smg_grpc_proto")
grpc = pytest.importorskip("grpc")
zmq = pytest.importorskip("zmq")
import zmq.asyncio  # noqa: E402, F811
from smg_grpc_proto.generated import common_pb2  # noqa: E402

_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang" / "kv_events.py"
_spec = importlib.util.spec_from_file_location("sglang_kv_transport", _PATH)
transport = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(transport)


def decode(payload):
    return int(payload)


def convert(value, seq):
    return common_pb2.KvEventBatch(sequence_number=seq, timestamp=value)


def subscribe_method():
    # Load the real RPC method without importing SGLang/torch. Its transport
    # and gRPC context are real; only engine batch decoding is substituted.
    path = _PATH.with_name("servicer.py")
    tree = ast.parse(path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "SubscribeKvEvents"
    )
    namespace = {
        "AsyncIterator": AsyncIterator,
        "common_pb2": common_pb2,
        "grpc": grpc,
        "asyncio": asyncio,
        "zmq": zmq,
        "logger": logging.getLogger(__name__),
        "KVEventBatch": object,
        "msgspec": SimpleNamespace(
            msgpack=SimpleNamespace(Decoder=lambda _: SimpleNamespace(decode=decode))
        ),
        "ZmqEventPublisher": SimpleNamespace(offset_endpoint_port=transport.endpoint_for_rank),
        "subscribe_kv_events": transport.subscribe_kv_events,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["SubscribeKvEvents"]


@pytest_asyncio.fixture
async def bridge():
    ctx = zmq.asyncio.Context()
    pub = ctx.socket(zmq.XPUB)
    pub.setsockopt(zmq.XPUB_VERBOSE, 1)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    config = SimpleNamespace(endpoint=f"tcp://127.0.0.1:{port}", topic="kv")
    cursors = []
    method = subscribe_method()
    servicer = SimpleNamespace(_kv_events_config=config, _convert_kv_event_batch=convert)

    async def handler(request, context):
        cursors.append(request.start_sequence_number)
        async for batch in method(servicer, request, context):
            yield batch

    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "test.KvEvents",
                {
                    "Subscribe": grpc.unary_stream_rpc_method_handler(
                        handler,
                        request_deserializer=common_pb2.SubscribeKvEventsRequest.FromString,
                        response_serializer=common_pb2.KvEventBatch.SerializeToString,
                    )
                },
            ),
        )
    )
    grpc_port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{grpc_port}")
    rpc = channel.unary_stream(
        "/test.KvEvents/Subscribe",
        request_serializer=common_pb2.SubscribeKvEventsRequest.SerializeToString,
        response_deserializer=common_pb2.KvEventBatch.FromString,
    )

    def subscribe(cursor=0):
        return rpc(common_pb2.SubscribeKvEventsRequest(start_sequence_number=cursor))

    async def subscribed():
        # XPUB acknowledges the actual subscription; no timing sleeps needed.
        while await asyncio.wait_for(pub.recv(), 3) != b"\x01kv":
            pass

    async def publish(seq, payload=None):
        await pub.send_multipart([b"kv", seq.to_bytes(8, "big"), payload or str(seq).encode()])

    try:
        yield SimpleNamespace(
            subscribe=subscribe,
            subscribed=subscribed,
            publish=publish,
            config=config,
            cursors=cursors,
            pub=pub,
            ctx=ctx,
        )
    finally:
        await channel.close()
        await server.stop(None)
        pub.close(linger=0)
        ctx.term()


async def read(call):
    return await asyncio.wait_for(call.read(), 3)


@pytest.mark.asyncio
async def test_gap_resume_is_rejected_then_live_cursor_advances(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(100)
    assert (await read(call)).sequence_number == 100
    await bridge.publish(103)
    assert (await read(call)).sequence_number == 103
    call.cancel()

    # The router retries from its last contiguous batch. Do not silently
    # reconnect to live seq=110 while it still expects seq=101.
    retry = bridge.subscribe(100)
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await read(retry)
    assert error.value.code() == grpc.StatusCode.OUT_OF_RANGE

    # OUT_OF_RANGE triggers the gateway's existing per-worker clear/reset.
    fresh = bridge.subscribe()
    await bridge.subscribed()
    for seq in (110, 111):
        await bridge.publish(seq)
        assert (await read(fresh)).sequence_number == seq
    assert bridge.cursors == [0, 100, 0]
    fresh.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [1, 2**64 - 1])
async def test_replay_rejected_before_opening_live_subscription(bridge, cursor):
    call = bridge.subscribe(cursor)
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await read(call)
    assert error.value.code() == grpc.StatusCode.OUT_OF_RANGE
    assert not await bridge.pub.poll(timeout=50)


@pytest.mark.asyncio
async def test_idle_poll_preserves_later_events_and_cancellation(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    await asyncio.wait_for(call.initial_metadata(), 3)
    await asyncio.sleep(1.1)  # exercise the idle poll timeout
    await bridge.publish(0)
    assert (await read(call)).sequence_number == 0
    call.cancel()
    assert await asyncio.wait_for(bridge.pub.recv(), 3) == b"\x00kv"


@pytest.mark.asyncio
async def test_bad_payload_does_not_hide_native_sequence_gap(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(10)
    assert (await read(call)).sequence_number == 10
    await bridge.pub.send_multipart([b"kv", b"short"])
    await bridge.publish(11, b"undecodable")
    await bridge.publish(12)
    assert (await read(call)).sequence_number == 12
    call.cancel()
