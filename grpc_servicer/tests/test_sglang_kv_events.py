"""The SGLang bridge's KV-event wiring: every DP rank's publisher goes through
the shared relay (its protocol has its own tests in ``test_kv_relay.py``)."""

import ast
import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
grpc = pytest.importorskip("grpc")
pytest.importorskip("zmq")
pytest.importorskip("msgspec")
from smg_grpc_proto.generated import common_pb2  # noqa: E402
from smg_grpc_servicer.sglang import kv_events as transport  # noqa: E402

_SERVICER = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang" / "servicer.py"


def test_one_source_per_data_parallel_rank():
    config = SimpleNamespace(endpoint="tcp://*:5557", replay_endpoint="tcp://*:5558", topic="")
    assert transport.dp_rank_count(SimpleNamespace(dp_size=3)) == 3
    assert transport.dp_rank_count(SimpleNamespace(dp_size=None)) == 1
    assert transport.dp_rank_count(SimpleNamespace()) == 1
    sources = transport.sources(config, SimpleNamespace(dp_size=2))
    assert [(s.rank, s.endpoint, s.replay_endpoint) for s in sources] == [
        (0, "tcp://127.0.0.1:5557", "tcp://127.0.0.1:5558"),
        (1, "tcp://127.0.0.1:5558", "tcp://127.0.0.1:5559"),
    ]
    without_replay = transport.sources(
        SimpleNamespace(endpoint="tcp://*:6000"), SimpleNamespace(dp_size=1)
    )
    assert without_replay[0].replay_endpoint is None


def _subscribe_method(namespace):
    # The real RPC method without importing SGLang/torch.
    tree = ast.parse(_SERVICER.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "SubscribeKvEvents"
    )
    namespace.update({"AsyncIterator": AsyncIterator, "common_pb2": common_pb2, "grpc": grpc})
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(_SERVICER), "exec"), namespace)
    return namespace["SubscribeKvEvents"]


@pytest.mark.asyncio
async def test_servicer_hands_config_server_args_and_cursor_to_the_relay():
    calls = []

    async def fake_subscribe(config, server_args, start, context):
        calls.append((config, server_args, start, context))
        yield common_pb2.KvEventBatch(sequence_number=1)

    method = _subscribe_method({"subscribe_kv_events": fake_subscribe})
    config = SimpleNamespace(endpoint="tcp://*:5557", topic="kv")
    server_args = SimpleNamespace(dp_size=2)
    servicer = SimpleNamespace(_kv_events_config=config, server_args=server_args)
    context = SimpleNamespace()
    request = common_pb2.SubscribeKvEventsRequest(start_sequence_number=7)
    batches = [batch async for batch in method(servicer, request, context)]
    assert [b.sequence_number for b in batches] == [1]
    assert calls == [(config, server_args, 7, context)]


@pytest.mark.asyncio
async def test_servicer_without_a_publisher_is_unimplemented():
    aborted = []

    class Context:
        async def abort(self, code, message):
            aborted.append((code, message))
            raise asyncio.CancelledError  # grpc's abort never returns

    method = _subscribe_method({"subscribe_kv_events": None})
    servicer = SimpleNamespace(_kv_events_config=None, server_args=SimpleNamespace(dp_size=1))
    with pytest.raises(asyncio.CancelledError):
        async for _ in method(servicer, common_pb2.SubscribeKvEventsRequest(), Context()):
            pass
    assert aborted[0][0] == grpc.StatusCode.UNIMPLEMENTED
    assert "--kv-events-config" in aborted[0][1]
