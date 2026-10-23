"""Unit tests for the TokenSpeed KV-events config resolver, its rank sources
and the shared conversion; the relay's protocol has its own tests in
``test_kv_relay.py``.

Engine-free: no TokenSpeed install required. Modules are loaded by file path so
the ``tokenspeed`` package ``__init__`` (which imports the engine-bound
servicer) is never triggered.

Run with: pytest grpc_servicer/tests/test_tokenspeed_kv_events.py -v
"""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
grpc = pytest.importorskip("grpc")
zmq = pytest.importorskip("zmq")
msgspec = pytest.importorskip("msgspec")
import zmq.asyncio  # noqa: E402, F811
from smg_grpc_proto.generated import common_pb2  # noqa: E402

_SERVICER_ROOT = Path(__file__).parents[1] / "smg_grpc_servicer"


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _SERVICER_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts_kv_events = _load("tokenspeed/kv_events.py", "ts_kv_events")
shared = _load("kv_events.py", "shared_kv_events")


def test_tokenspeed_servicer_relays_through_the_shared_relay():
    """The servicer opens no SUB socket of its own: every rank's publisher
    goes through the transport module and the shared relay."""
    source = (_SERVICER_ROOT / "tokenspeed" / "servicer.py").read_text()

    assert "subscribe_kv_events," in source
    assert "zmq.SUB" not in source and "stream_kv_events" not in source
    assert "tokenspeed.runtime.disaggregation.kv.kv_events" not in source


class _Args:
    """Minimal stand-in for TokenSpeed's ServerArgs."""

    def __init__(self, kv_events_config):
        self.kv_events_config = kv_events_config


def _cfg(**kw) -> str:
    return json.dumps(kw)


class TestResolveKvEventsConfig:
    def test_none_when_attr_missing(self):
        assert ts_kv_events.resolve_kv_events_config(object()) is None

    def test_none_when_empty_string(self):
        assert ts_kv_events.resolve_kv_events_config(_Args("")) is None

    def test_none_when_none(self):
        assert ts_kv_events.resolve_kv_events_config(_Args(None)) is None

    def test_none_when_disabled(self):
        cfg = _cfg(enable_kv_cache_events=False, publisher="zmq")
        assert ts_kv_events.resolve_kv_events_config(_Args(cfg)) is None

    def test_none_when_publisher_non_zmq_string(self):
        # Explicit non-zmq publisher (string "null") disables bridging.
        cfg = _cfg(enable_kv_cache_events=True, publisher="null")
        assert ts_kv_events.resolve_kv_events_config(_Args(cfg)) is None

    def test_returns_config_when_enabled_zmq(self):
        cfg = _cfg(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint="tcp://*:6000",
            topic="kv",
        )
        out = ts_kv_events.resolve_kv_events_config(_Args(cfg))
        assert out is not None
        assert out.endpoint == "tcp://*:6000"
        assert out.topic == "kv"

    def test_publisher_defaults_to_zmq_when_enabled(self):
        # publisher unset → "zmq" when enabled (matches EventPublisherFactory).
        out = ts_kv_events.resolve_kv_events_config(_Args(_cfg(enable_kv_cache_events=True)))
        assert out is not None
        assert out.endpoint == "tcp://*:5557"  # KVEventsConfig default
        assert out.topic == ""  # KVEventsConfig default

    def test_publisher_explicit_json_null_defaults_to_zmq(self):
        # Explicit JSON ``null`` (not the string "null") → defaults to zmq when enabled.
        cfg = _cfg(enable_kv_cache_events=True, publisher=None)
        assert '"publisher": null' in cfg
        out = ts_kv_events.resolve_kv_events_config(_Args(cfg))
        assert out is not None

    def test_none_when_endpoint_or_topic_not_string(self):
        # Non-string endpoint/topic from arbitrary JSON → cleanly disabled (not a runtime crash).
        assert (
            ts_kv_events.resolve_kv_events_config(
                _Args(_cfg(enable_kv_cache_events=True, endpoint=1234))
            )
            is None
        )
        assert (
            ts_kv_events.resolve_kv_events_config(
                _Args(_cfg(enable_kv_cache_events=True, topic=["x"]))
            )
            is None
        )

    def test_malformed_json_returns_none(self):
        assert ts_kv_events.resolve_kv_events_config(_Args("{not json")) is None

    def test_non_object_json_returns_none(self):
        assert ts_kv_events.resolve_kv_events_config(_Args("[1, 2, 3]")) is None


# --- TokenSpeed-shaped fake events (dispatch is by class name) ---
# TokenSpeed's BlockStored has no lora_id; its batch carries attn_dp_rank.
class BlockStored:
    def __init__(self, block_hashes, parent_block_hash, token_ids, block_size):
        self.block_hashes = block_hashes
        self.parent_block_hash = parent_block_hash
        self.token_ids = token_ids
        self.block_size = block_size


class BlockRemoved:
    def __init__(self, block_hashes):
        self.block_hashes = block_hashes


class AllBlocksCleared:
    pass


class KVEventBatch:
    def __init__(self, ts, events, attn_dp_rank=None):
        self.ts = ts
        self.events = events
        self.attn_dp_rank = attn_dp_rank


class TestConvertEventTokenSpeedShape:
    def test_block_stored_without_lora(self):
        ev = BlockStored(
            block_hashes=[111], parent_block_hash=None, token_ids=[1, 2, 3, 4], block_size=4
        )
        out = shared.convert_event(ev, event_id=1)
        assert out.WhichOneof("data") == "stored"
        assert out.stored.blocks[0].block_hash == 111
        assert list(out.stored.blocks[0].token_ids) == [1, 2, 3, 4]
        assert not out.stored.blocks[0].HasField("lora_id")
        assert not out.stored.HasField("parent_block_hash")

    def test_block_stored_multi_block_with_parent(self):
        ev = BlockStored(
            block_hashes=[10, 20],
            parent_block_hash=9,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            block_size=4,
        )
        out = shared.convert_event(ev, event_id=1)
        assert [b.block_hash for b in out.stored.blocks] == [10, 20]
        assert list(out.stored.blocks[1].token_ids) == [5, 6, 7, 8]
        assert out.stored.parent_block_hash == 9

    def test_block_removed(self):
        out = shared.convert_event(BlockRemoved(block_hashes=[1, 2]), event_id=2)
        assert out.WhichOneof("data") == "removed"
        assert list(out.removed.block_hashes) == [1, 2]

    def test_all_blocks_cleared(self):
        out = shared.convert_event(AllBlocksCleared(), event_id=3)
        assert out.WhichOneof("data") == "cleared"


class TestConvertBatchAttnDpRank:
    def test_attn_dp_rank_maps_to_proto_dp_rank(self):
        batch = KVEventBatch(ts=1.0, events=[], attn_dp_rank=2)
        proto, _ = shared.convert_batch(batch, seq_num=5, event_id_start=0)
        assert proto.sequence_number == 5
        assert proto.dp_rank == 2

    def test_no_dp_rank_when_attn_dp_rank_none(self):
        batch = KVEventBatch(ts=1.0, events=[], attn_dp_rank=None)
        proto, _ = shared.convert_batch(batch, seq_num=1, event_id_start=0)
        assert not proto.HasField("dp_rank")

    def test_attn_dp_rank_zero_is_set(self):
        # rank 0 is a valid rank, not "unset" — it must still be encoded.
        batch = KVEventBatch(ts=1.0, events=[], attn_dp_rank=0)
        proto, _ = shared.convert_batch(batch, seq_num=1, event_id_start=0)
        assert proto.HasField("dp_rank")
        assert proto.dp_rank == 0


def _dp_args(dp_size):
    return SimpleNamespace(mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=dp_size)))


def test_one_source_per_attention_dp_rank():
    """A worker with attention data parallelism publishes one stream per rank
    on ``port + rank`` (its replay socket likewise): one source each."""
    assert ts_kv_events.dp_rank_count(_dp_args(3)) == 3
    assert ts_kv_events.dp_rank_count(_dp_args(None)) == 1
    assert ts_kv_events.dp_rank_count(SimpleNamespace()) == 1
    config = ts_kv_events.ResolvedKvEventsConfig(
        endpoint="tcp://*:5557", topic="", replay_endpoint="tcp://*:5600"
    )
    sources = ts_kv_events.sources(config, _dp_args(2))
    assert [(s.rank, s.endpoint, s.replay_endpoint) for s in sources] == [
        (0, "tcp://127.0.0.1:5557", "tcp://127.0.0.1:5600"),
        (1, "tcp://127.0.0.1:5558", "tcp://127.0.0.1:5601"),
    ]
    without_replay = ts_kv_events.sources(
        ts_kv_events.ResolvedKvEventsConfig(endpoint="tcp://*:6000", topic=""), _dp_args(1)
    )
    assert [(s.rank, s.endpoint, s.replay_endpoint) for s in without_replay] == [
        (0, "tcp://127.0.0.1:6000", None)
    ]


class TestResolvedReplayEndpoint:
    def test_replay_endpoint_is_carried_when_configured(self):
        cfg = ts_kv_events.resolve_kv_events_config(
            _Args(
                _cfg(
                    enable_kv_cache_events=True,
                    publisher="zmq",
                    endpoint="tcp://*:5600",
                    replay_endpoint="tcp://*:5601",
                )
            )
        )
        assert (cfg.endpoint, cfg.replay_endpoint) == ("tcp://*:5600", "tcp://*:5601")

    def test_replay_endpoint_is_empty_when_absent_or_not_a_string(self):
        absent = ts_kv_events.resolve_kv_events_config(
            _Args(_cfg(enable_kv_cache_events=True, publisher="zmq"))
        )
        assert absent.replay_endpoint == ""
        odd = ts_kv_events.resolve_kv_events_config(
            _Args(_cfg(enable_kv_cache_events=True, publisher="zmq", replay_endpoint=5601))
        )
        assert odd is not None and odd.replay_endpoint == ""


# --- The servicer hands its load record to the stream loop (no TokenSpeed import) ---


def _subscribe_method(namespace):
    """The real ``SubscribeKvEvents`` method of the TokenSpeed servicer,
    compiled on its own (the servicer module imports the engine)."""
    import ast
    import logging
    from collections.abc import AsyncIterator

    source_path = _SERVICER_ROOT / "tokenspeed" / "servicer.py"
    tree = ast.parse(source_path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "SubscribeKvEvents"
    )
    namespace.update(
        {
            "AsyncIterator": AsyncIterator,
            "asyncio": asyncio,
            "common_pb2": common_pb2,
            "grpc": grpc,
            "logger": logging.getLogger("test"),
        }
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["SubscribeKvEvents"]


@pytest.mark.asyncio
async def test_servicer_hands_config_server_args_and_cursor_to_the_relay():
    calls = []

    async def fake_subscribe(config, server_args, start, context, *, load_source=None):
        calls.append((config, server_args, start, context, load_source))
        yield common_pb2.KvEventBatch(sequence_number=1)

    method = _subscribe_method({"subscribe_kv_events": fake_subscribe})
    config = SimpleNamespace(endpoint="tcp://*:5557", topic="kv")
    server_args = _dp_args(2)

    def load_record(dp_rank):
        return None

    servicer = SimpleNamespace(
        _kv_events_config=config, server_args=server_args, _load_record=load_record
    )
    context = SimpleNamespace()
    request = common_pb2.SubscribeKvEventsRequest(start_sequence_number=7)
    batches = [batch async for batch in method(servicer, request, context)]
    assert [b.sequence_number for b in batches] == [1]
    # The servicer's load record rides the stream: its source goes to the relay.
    assert calls == [(config, server_args, 7, context, load_record)]


@pytest.mark.asyncio
async def test_servicer_without_a_publisher_is_unimplemented():
    aborted = []

    class Context:
        async def abort(self, code, message):
            aborted.append((code, message))
            raise asyncio.CancelledError  # grpc's abort never returns

    method = _subscribe_method({"subscribe_kv_events": None})
    servicer = SimpleNamespace(_kv_events_config=None, server_args=_dp_args(1))
    with pytest.raises(asyncio.CancelledError):
        async for _ in method(servicer, common_pb2.SubscribeKvEventsRequest(), Context()):
            pass
    assert aborted[0][0] == grpc.StatusCode.UNIMPLEMENTED
    assert "--kv-events-config" in aborted[0][1]


# --- Two ranks, two endpoints, through the transport module -----------------


# The engine's wire structs (array_like + tagged union), as the publisher
# encodes them; the tag is the engine's class name.
class WireStored(msgspec.Struct, array_like=True, omit_defaults=True, tag="BlockStored"):
    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
    block_size: int


class WireBatch(msgspec.Struct, array_like=True, omit_defaults=True):
    ts: float
    events: list[WireStored]
    attn_dp_rank: int | None = None


def _bind_consecutive(ctx, count, attempts=20):
    """``count`` XPUB sockets on consecutive ports, as the engine lays out its ranks."""
    for _ in range(attempts):
        sockets = []
        try:
            first = ctx.socket(zmq.XPUB)
            first.setsockopt(zmq.XPUB_VERBOSE, 1)
            base = first.bind_to_random_port("tcp://127.0.0.1")
            sockets.append(first)
            for offset in range(1, count):
                sock = ctx.socket(zmq.XPUB)
                sock.setsockopt(zmq.XPUB_VERBOSE, 1)
                sock.bind(f"tcp://127.0.0.1:{base + offset}")
                sockets.append(sock)
            return base, sockets
        except zmq.ZMQError:
            for sock in sockets:
                sock.close(linger=0)
    raise RuntimeError("no consecutive ports available")


class _RpcContext:
    """What the relay asks of the gRPC context: headers, cancellation, abort."""

    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.delivered = 0
        self.aborted = None

    def cancelled(self):
        return self.delivered >= self.stop_after

    async def send_initial_metadata(self, metadata):
        pass

    async def abort(self, code, message):
        self.aborted = (code, message)
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_a_two_rank_worker_is_relayed_from_both_endpoints():
    """A two-rank configuration subscribes to two endpoints (``port`` and
    ``port + 1``), and the stream carries both ranks' batches, each tagged
    with its publisher's rank."""
    ctx = zmq.asyncio.Context()
    base, pubs = _bind_consecutive(ctx, 2)
    config = ts_kv_events.ResolvedKvEventsConfig(endpoint=f"tcp://127.0.0.1:{base}", topic="kv")
    encoder = msgspec.msgpack.Encoder()
    rpc = _RpcContext(stop_after=2)
    seen = []
    try:

        async def consume():
            async for batch in ts_kv_events.subscribe_kv_events(config, _dp_args(2), 0, rpc):
                seen.append(batch)
                rpc.delivered += 1

        consumer = asyncio.create_task(consume())
        # XPUB acknowledges each actual subscription: both ranks were connected.
        for pub in pubs:
            while await asyncio.wait_for(pub.recv(), 5) != b"\x01kv":
                pass
        for rank, pub in enumerate(pubs):
            payload = encoder.encode(
                WireBatch(
                    ts=1.0,
                    events=[
                        WireStored(
                            block_hashes=[100 + rank],
                            parent_block_hash=None,
                            token_ids=[1, 2],
                            block_size=2,
                        )
                    ],
                    attn_dp_rank=rank,
                )
            )
            await pub.send_multipart([b"kv", (0).to_bytes(8, "big"), payload])
        await asyncio.wait_for(consumer, timeout=10)
    finally:
        for pub in pubs:
            pub.close(linger=0)
        ctx.term()

    assert rpc.aborted is None
    by_rank = {batch.dp_rank: batch.events[0].stored.blocks[0].block_hash for batch in seen}
    assert by_rank == {0: 100, 1: 101}
