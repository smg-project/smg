"""Unit tests for vLLM KV-event conversion helpers (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_vllm_kv_events.py -v
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("smg_grpc_proto")

# Import the module directly to avoid pulling vllm via the package __init__.
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "kv_events.py"
_spec = importlib.util.spec_from_file_location("vllm_kv_events", _MODULE_PATH)
kv_events = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kv_events)


# --- Fake vLLM event objects (dispatch is by class name, so names matter) ---
class BlockStored:
    def __init__(self, block_hashes, parent_block_hash, token_ids, block_size, lora_id=None):
        self.block_hashes = block_hashes
        self.parent_block_hash = parent_block_hash
        self.token_ids = token_ids
        self.block_size = block_size
        self.lora_id = lora_id


class BlockRemoved:
    def __init__(self, block_hashes):
        self.block_hashes = block_hashes


class AllBlocksCleared:
    pass


class KVEventBatch:
    def __init__(self, ts, events, data_parallel_rank=None):
        self.ts = ts
        self.events = events
        self.data_parallel_rank = data_parallel_rank


class TestToInt64:
    def test_small_positive_unchanged(self):
        assert kv_events.to_int64(42) == 42

    def test_value_above_i64_max_wraps_to_negative(self):
        # 2**63 wraps to the minimum signed 64-bit value.
        assert kv_events.to_int64(2**63) == -(2**63)

    def test_full_256bit_hash_reduced_deterministically(self):
        h = 0xDEADBEEFCAFEBABE_1122334455667788  # 128-bit
        out = kv_events.to_int64(h)
        assert -(2**63) <= out < 2**63
        assert out == kv_events.to_int64(h)  # deterministic
        # Equals the low 64 bits, reinterpreted as signed.
        low = h & 0xFFFFFFFFFFFFFFFF
        assert out == (low - 2**64 if low >= 2**63 else low)

    def test_bytes_hash_interpreted_big_endian(self):
        assert kv_events.to_int64((1).to_bytes(8, "big")) == 1
        assert kv_events.to_int64(b"\x00" * 8) == 0

    def test_bytes_hash_wraps_to_int64_like_int_path(self):
        # bytes path must produce the same identity as the equivalent int.
        wide = 0xDEADBEEFCAFEBABE
        assert kv_events.to_int64(wide.to_bytes(8, "big")) == kv_events.to_int64(wide)
        assert kv_events.to_int64((2**63).to_bytes(8, "big")) == -(2**63)


class TestEndpointForRank:
    def test_bind_star_becomes_loopback(self):
        assert kv_events.endpoint_for_rank("tcp://*:5557", 0) == "tcp://127.0.0.1:5557"

    def test_rank_offsets_port(self):
        assert kv_events.endpoint_for_rank("tcp://*:5557", 3) == "tcp://127.0.0.1:5560"

    def test_explicit_host_preserved(self):
        assert kv_events.endpoint_for_rank("tcp://10.0.0.2:6000", 0) == "tcp://10.0.0.2:6000"

    def test_ipc_endpoint_passthrough(self):
        assert kv_events.endpoint_for_rank("ipc:///tmp/kv", 0) == "ipc:///tmp/kv"

    def test_portless_tcp_wildcard_no_crash(self):
        assert kv_events.endpoint_for_rank("tcp://*", 2) == "tcp://127.0.0.1"

    def test_zero_host_becomes_loopback(self):
        # 0.0.0.0 is not connectable on macOS/Windows; must rewrite to 127.0.0.1.
        assert kv_events.endpoint_for_rank("tcp://0.0.0.0:5557", 0) == "tcp://127.0.0.1:5557"
        assert kv_events.endpoint_for_rank("tcp://0.0.0.0:5557", 2) == "tcp://127.0.0.1:5559"


class TestResolveKvEventsConfig:
    class _Cfg:
        def __init__(self, enable, publisher, endpoint="tcp://*:5557", topic=""):
            self.enable_kv_cache_events = enable
            self.publisher = publisher
            self.endpoint = endpoint
            self.topic = topic

    class _Engine:
        def __init__(self, cfg):
            self.vllm_config = type("VC", (), {"kv_events_config": cfg})()

    def test_none_when_no_vllm_config(self):
        engine = type("E", (), {})()  # no vllm_config attr
        assert kv_events.resolve_kv_events_config(engine) is None

    def test_none_when_disabled(self):
        engine = self._Engine(self._Cfg(enable=False, publisher="zmq"))
        assert kv_events.resolve_kv_events_config(engine) is None

    def test_none_when_publisher_not_zmq(self):
        engine = self._Engine(self._Cfg(enable=True, publisher="null"))
        assert kv_events.resolve_kv_events_config(engine) is None

    def test_returns_cfg_when_enabled_zmq(self):
        cfg = self._Cfg(enable=True, publisher="zmq")
        engine = self._Engine(cfg)
        assert kv_events.resolve_kv_events_config(engine) is cfg


class TestConvertEvent:
    @pytest.mark.parametrize(
        "hashes,tokens,block_size",
        [
            ([4, 8, 10, 12], list(range(1, 13)), 2),  # Leading and interior holes.
            ([2, 6], list(range(1, 7)), 2),  # Interior hole.
            ([1, 2], [1, 2], 2),  # More hashes than token blocks.
            ([1], [1, 2, 3], 2),  # Partial token block.
            ([1], [], 0),
            ([1], [], -1),
        ],
    )
    def test_unaligned_store_is_skipped(self, hashes, tokens, block_size, caplog):
        ev = BlockStored(hashes, None, tokens, block_size)
        assert kv_events.convert_event(ev, event_id=7) is None
        assert "Skipping BlockStored" in caplog.text

    def test_block_stored_single_block(self):
        ev = BlockStored(
            block_hashes=[111], parent_block_hash=None, token_ids=[1, 2, 3, 4], block_size=4
        )
        out = kv_events.convert_event(ev, event_id=7)
        assert out.event_id == 7
        assert out.WhichOneof("data") == "stored"
        assert len(out.stored.blocks) == 1
        assert out.stored.blocks[0].block_hash == 111
        assert list(out.stored.blocks[0].token_ids) == [1, 2, 3, 4]
        assert out.stored.blocks[0].block_size == 4
        assert not out.stored.HasField("parent_block_hash")

    def test_block_stored_multi_block_slices_tokens(self):
        ev = BlockStored(
            block_hashes=[10, 20],
            parent_block_hash=9,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            block_size=4,
        )
        out = kv_events.convert_event(ev, event_id=1)
        assert [b.block_hash for b in out.stored.blocks] == [10, 20]
        assert list(out.stored.blocks[0].token_ids) == [1, 2, 3, 4]
        assert list(out.stored.blocks[1].token_ids) == [5, 6, 7, 8]
        assert out.stored.parent_block_hash == 9

    def test_block_stored_with_lora(self):
        ev = BlockStored(
            block_hashes=[1], parent_block_hash=None, token_ids=[1, 2], block_size=2, lora_id=5
        )
        out = kv_events.convert_event(ev, event_id=1)
        assert out.stored.blocks[0].lora_id == 5

    def test_block_stored_wide_hashes_reduced_to_int64(self):
        ev = BlockStored(
            block_hashes=[2**63],
            parent_block_hash=2**63,
            token_ids=[1, 2],
            block_size=2,
            lora_id=2**63,
        )
        out = kv_events.convert_event(ev, event_id=1)
        assert out.stored.blocks[0].block_hash == -(2**63)
        assert out.stored.parent_block_hash == -(2**63)
        assert out.stored.blocks[0].lora_id == -(2**63)

    def test_block_stored_bytes_block_hash(self):
        ev = BlockStored(
            block_hashes=[(5).to_bytes(8, "big")],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
        )
        out = kv_events.convert_event(ev, event_id=1)
        assert out.stored.blocks[0].block_hash == 5

    def test_block_removed(self):
        out = kv_events.convert_event(BlockRemoved(block_hashes=[1, 2, 3]), event_id=2)
        assert out.WhichOneof("data") == "removed"
        assert list(out.removed.block_hashes) == [1, 2, 3]

    def test_all_blocks_cleared(self):
        out = kv_events.convert_event(AllBlocksCleared(), event_id=3)
        assert out.WhichOneof("data") == "cleared"

    def test_unknown_event_returns_none(self):
        class Mystery:
            pass

        assert kv_events.convert_event(Mystery(), event_id=4) is None


class TestConvertBatch:
    def test_skipped_store_preserves_batch_and_later_events(self):
        batch = KVEventBatch(
            ts=12.5,
            events=[
                BlockStored([4, 8, 10, 12], None, list(range(1, 13)), 2),
                BlockStored([14], None, [13, 14], 2),
            ],
            data_parallel_rank=2,
        )
        proto, next_id = kv_events.convert_batch(batch, seq_num=99, event_id_start=10)
        assert (proto.sequence_number, proto.timestamp, proto.dp_rank) == (99, 12.5, 2)
        assert [e.event_id for e in proto.events] == [12]
        assert next_id == 12
        block = proto.events[0].stored.blocks[0]
        assert (block.block_hash, list(block.token_ids), block.block_size) == (14, [13, 14], 2)

    def test_seq_timestamp_and_event_ids(self):
        batch = KVEventBatch(
            ts=12.5,
            events=[
                BlockStored(
                    block_hashes=[1], parent_block_hash=None, token_ids=[1, 2], block_size=2
                ),
                BlockRemoved(block_hashes=[1]),
            ],
        )
        proto, next_id = kv_events.convert_batch(batch, seq_num=99, event_id_start=0)
        assert proto.sequence_number == 99
        assert abs(proto.timestamp - 12.5) < 1e-9
        assert [e.event_id for e in proto.events] == [1, 2]
        assert next_id == 2
        assert not proto.HasField("dp_rank")

    def test_dp_rank_set_when_present(self):
        batch = KVEventBatch(ts=1.0, events=[], data_parallel_rank=2)
        proto, _ = kv_events.convert_batch(batch, seq_num=1, event_id_start=0)
        assert proto.dp_rank == 2

    def test_unknown_events_skipped_but_counter_advances(self):
        class Mystery:
            pass

        batch = KVEventBatch(ts=1.0, events=[Mystery(), BlockRemoved(block_hashes=[5])])
        proto, next_id = kv_events.convert_batch(batch, seq_num=1, event_id_start=10)
        # One event emitted (the BlockRemoved), but the id counter advanced past both.
        assert len(proto.events) == 1
        assert proto.events[0].event_id == 12
        assert next_id == 12


class TestServicerWiring:
    def test_servicer_resolves_config_from_engine(self):
        pytest.importorskip("vllm")
        from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

        class _Cfg:
            enable_kv_cache_events = True
            publisher = "zmq"
            endpoint = "tcp://*:5557"
            topic = ""

        class _Engine:
            vllm_config = type("VC", (), {"kv_events_config": _Cfg()})()

        servicer = VllmEngineServicer(_Engine(), start_time=0.0)
        assert servicer._kv_events_config is not None
        assert hasattr(servicer, "SubscribeKvEvents")


# --- The servicer hands its load record to the relay (no vLLM import) ---


def _subscribe_method(namespace):
    """The real ``SubscribeKvEvents`` method of the vLLM servicer, compiled on
    its own (the servicer module imports the engine)."""
    import ast
    from collections.abc import AsyncIterator

    import grpc
    from smg_grpc_proto.generated import common_pb2

    source_path = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "servicer.py"
    tree = ast.parse(source_path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "SubscribeKvEvents"
    )
    namespace.update({"AsyncIterator": AsyncIterator, "common_pb2": common_pb2, "grpc": grpc})
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["SubscribeKvEvents"]


@pytest.mark.asyncio
async def test_servicer_hands_its_load_record_to_the_relay():
    from types import SimpleNamespace

    from smg_grpc_proto.generated import common_pb2

    calls = []

    async def fake_relay(sources, engine, start, context, **options):
        calls.append((sources, engine, start, context, options))
        yield common_pb2.KvEventBatch(sequence_number=1)

    method = _subscribe_method(
        {
            "relay": fake_relay,
            "rank_sources_for": lambda config, engine: ["rank-0"],
            "Engine": SimpleNamespace(VLLM="vllm"),
        }
    )
    config = SimpleNamespace(endpoint="tcp://*:5557", topic="kv", hwm=None)

    def load_record(dp_rank):
        return None

    servicer = SimpleNamespace(_kv_events_config=config, engine=object(), _load_record=load_record)
    context = SimpleNamespace()
    request = common_pb2.SubscribeKvEventsRequest(start_sequence_number=0)
    batches = [batch async for batch in method(servicer, request, context)]
    assert [b.sequence_number for b in batches] == [1]
    ((sources, engine, start, got_context, options),) = calls
    assert (sources, engine, start, got_context) == (["rank-0"], "vllm", 0, context)
    assert options["topic"] == "kv" and options["load_source"] is load_record


# --- The servicer's record is the batch's rank's own (no vLLM import) ---


def _load_record_method():
    """The real ``_load_record`` of the vLLM servicer with the readers it
    calls (``_scheduler_counts``, ``_latest_scheduler_stats``, ``_first_rank``),
    compiled on their own (the servicer module imports the engine), and the
    loads module they draw on, loaded by path for the same reason."""
    import ast
    import logging

    from smg_grpc_proto import vllm_engine_pb2
    from smg_grpc_proto.generated import common_pb2
    from smg_grpc_servicer.vllm.config_view import EngineConfigView

    package = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm"
    spec = importlib.util.spec_from_file_location("vllm_loads_for_the_record", package / "loads.py")
    loads_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loads_mod)
    source_path = package / "servicer.py"
    wanted = {
        "_latest_scheduler_stats",
        "_scheduler_counts",
        "_rank_counts",
        "_scheduler_loads",
        "_load_record",
        "_managed_ranks",
        "_first_rank",
        "_rank_in_flight",
        "_routed_rank",
        "_request_rank",
    }
    defs = [
        node
        for node in ast.walk(ast.parse(source_path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in defs} == wanted
    namespace = {
        "EngineConfigView": EngineConfigView,
        "common_pb2": common_pb2,
        "vllm_engine_pb2": vllm_engine_pb2,
        "load_record": loads_mod.load_record,
        "scheduler_load_fields": loads_mod.scheduler_load_fields,
        "logger": logging.getLogger("vllm_servicer_under_test"),
        "_kv_capacity_tokens": lambda engine: 0,
        "_max_running_requests": lambda engine: 0,
    }
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(source_path), "exec"), namespace)
    return loads_mod, namespace


def _servicer_over(stat_logger, managed=None):
    """A servicer whose engine keeps its scheduler stats on ``stat_logger``
    (and whose engine client manages the ranks ``managed``, when given)."""
    from types import MethodType, SimpleNamespace

    loads_mod, namespace = _load_record_method()
    engine = SimpleNamespace(logger_manager=SimpleNamespace(stat_loggers=[stat_logger]))
    if managed is not None:
        engine.engine_core = SimpleNamespace(engine_ranks_managed=managed)
    servicer = SimpleNamespace(engine=engine, _loads=loads_mod.LoadTracker())
    servicer._scheduler_counts = MethodType(namespace["_scheduler_counts"], servicer)
    servicer._rank_counts = MethodType(namespace["_rank_counts"], servicer)
    servicer._scheduler_loads = MethodType(namespace["_scheduler_loads"], servicer)
    servicer._load_record = MethodType(namespace["_load_record"], servicer)
    servicer._first_rank = MethodType(lambda self: namespace["_first_rank"](self.engine), servicer)
    return servicer


def _stats(running, waiting, usage):
    from types import SimpleNamespace

    return SimpleNamespace(num_running_reqs=running, num_waiting_reqs=waiting, kv_cache_usage=usage)


def _core(record):
    return record.running_requests, record.waiting_requests, record.token_usage


@pytest.mark.parametrize("shape", ["aggregated_logger", "per_engine_loggers"])
def test_the_record_is_the_batch_ranks_own_with_that_ranks_estimate(shape):
    from types import SimpleNamespace

    first, second = _stats(3, 2, 0.25), _stats(5, 1, 0.5)
    if shape == "aggregated_logger":
        stat_logger = SimpleNamespace(last_scheduler_stats_dict={0: first, 1: second})
    else:
        stat_logger = SimpleNamespace(
            per_engine_stat_loggers={
                0: SimpleNamespace(last_scheduler_stats=first),
                1: SimpleNamespace(last_scheduler_stats=second),
            }
        )
    servicer = _servicer_over(stat_logger)
    # Rank 1 served a request whose prompt it had cached and streamed 200
    # tokens; rank 0 served nothing yet.
    servicer._loads.submitted("r", 100, rank=1)
    servicer._loads.first_output("r", 100, 100, rank=1)
    servicer._loads.generated(200, rank=1)

    rank_one = servicer._load_record(1)
    assert _core(rank_one) == (5, 1, 0.5)
    # Rank 1's own figures ride its record, where the servicer's whole-pod
    # figures used to ride the first rank's and rank 1 read zero.
    assert rank_one.HasField("waiting_uncached_tokens")
    assert rank_one.cache_hit_rate == 1.0 and rank_one.gen_throughput == 100.0
    rank_zero = servicer._load_record(0)
    assert _core(rank_zero) == (3, 2, 0.25)
    assert rank_zero.HasField("waiting_uncached_tokens")
    assert (rank_zero.cache_hit_rate, rank_zero.gen_throughput) == (0.0, 0.0)
    assert servicer._load_record(None) == rank_zero


def test_a_request_is_credited_to_the_rank_the_engine_routed_it_to():
    """The rank of a request, for the tracker: the engine client's own
    in-flight table, keyed by the engine's internal request id (the output
    processor maps the caller's id to it) and valued with the two-byte
    identity that is the rank; else the pin the request named; else the
    servicer's first rank."""
    from types import SimpleNamespace

    _, namespace = _load_record_method()
    request_rank, routed_rank = namespace["_request_rank"], namespace["_routed_rank"]
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            reqs_in_flight={"routed-ab12cd34": (1).to_bytes(2, "little")},
            engine_ranks_managed=[0, 1],
        ),
        output_processor=SimpleNamespace(external_req_ids={"routed": ["routed-ab12cd34"]}),
    )
    assert request_rank(engine, "routed", None) == 1
    # The table wins over the pin (the client lists a pinned request too).
    assert request_rank(engine, "routed", 0) == 1
    # The caller's id is not a key of the table: the map is what finds it.
    engine.output_processor.external_req_ids = {}
    assert request_rank(engine, "routed", None) == 0
    # A request the table no longer lists: the pin, else the first rank.
    assert request_rank(engine, "finished", 1) == 1
    assert request_rank(engine, "finished", None) == 0
    # Each choice of an n > 1 request is routed on its own; its id starts
    # with its index.
    engine.output_processor.external_req_ids = {"fan": ["0_fan-ab12cd34", "1_fan-ab12cd34"]}
    engine.engine_core.reqs_in_flight = {
        "0_fan-ab12cd34": (0).to_bytes(2, "little"),
        "1_fan-ab12cd34": (1).to_bytes(2, "little"),
    }
    assert (request_rank(engine, "fan", None, 0), request_rank(engine, "fan", None, 1)) == (0, 1)
    # A choice already retired reads as its sibling's rank rather than none.
    del engine.engine_core.reqs_in_flight["0_fan-ab12cd34"]
    assert request_rank(engine, "fan", None, 0) == 1
    # A request the engine has retired from both tables (a non-streamed
    # request's only output is its last): what the tracker learned while
    # it was queued, before the pin and the first rank.
    engine.engine_core.reqs_in_flight = {}
    engine.output_processor.external_req_ids = {}
    assert request_rank(engine, "gone", None, 0, remembered=1) == 1
    assert request_rank(engine, "gone", 0, 0, remembered=1) == 1
    assert request_rank(engine, "gone", None, 0) == 0
    # Not routed yet: its queued prompt counts nowhere; a client without a
    # table (one engine) routes everything to the first rank.
    assert routed_rank(engine, "pending") is None
    one_engine = SimpleNamespace(engine_core=SimpleNamespace())
    assert routed_rank(one_engine, "r") == 0
    assert request_rank(one_engine, "r", None) == 0
    # A pin cannot name another rank on a client with one engine.
    assert request_rank(SimpleNamespace(), "r", 3) == 0


def test_a_rank_the_stat_loggers_do_not_keep_apart_gets_no_record():
    from types import SimpleNamespace

    servicer = _servicer_over(SimpleNamespace(last_scheduler_stats=_stats(3, 2, 0.25)))
    assert servicer._load_record(0).running_requests == 3
    assert servicer._load_record(1) is None


def test_the_first_rank_alone_reports_zeros_before_the_first_snapshot():
    from types import SimpleNamespace

    servicer = _servicer_over(SimpleNamespace())
    assert servicer._load_record(None).running_requests == 0
    assert servicer._load_record(1) is None


def test_a_servicer_managing_a_slice_of_the_ranks_counts_from_its_first_rank():
    from types import SimpleNamespace

    # External or hybrid load balancing: this servicer's engines are ranks 4 and
    # 5, the stat loggers are keyed by those global ranks, and the aggregated
    # logger's own snapshot is the aggregate (index -1), no rank's own.
    stat_logger = SimpleNamespace(
        engine_index=-1,
        last_scheduler_stats=_stats(9, 9, 0.9),
        last_scheduler_stats_dict={4: _stats(3, 2, 0.25), 5: _stats(5, 1, 0.5)},
    )
    servicer = _servicer_over(stat_logger, managed=[4, 5])

    assert servicer._first_rank() == 4
    # What GetLoads reads: the first rank's own counts, not the aggregate.
    assert servicer._scheduler_counts(servicer._first_rank()) == (3, 2, 0.25)
    first = servicer._load_record(None)
    assert _core(first) == (3, 2, 0.25) and first.HasField("waiting_uncached_tokens")
    assert servicer._load_record(4) == first
    assert _core(servicer._load_record(5)) == (5, 1, 0.5)
    assert servicer._load_record(5).HasField("waiting_uncached_tokens")
    assert servicer._load_record(0) is None


def test_get_loads_carries_every_rank_the_servicer_serves_with_its_own_figures():
    """A poll answers one entry per served rank, each the rank's own counts
    and estimate, the figures its pushed record carries: a gateway whose
    poll stands in for a quiet rank's records has every rank to refresh."""
    from types import SimpleNamespace

    stat_logger = SimpleNamespace(
        last_scheduler_stats_dict={0: _stats(3, 2, 0.25), 1: _stats(5, 1, 0.5)}
    )
    servicer = _servicer_over(stat_logger, managed=[0, 1])
    servicer._loads.submitted("r", 100, rank=1)
    servicer._loads.first_output("r", 100, 100, rank=1)
    servicer._loads.generated(200, rank=1)

    loads = servicer._scheduler_loads()
    assert [
        (load.dp_rank, load.num_running_reqs, load.num_waiting_reqs, load.token_usage)
        for load in loads
    ] == [(0, 3, 2, 0.25), (1, 5, 1, 0.5)]
    # Rank 1's own hit rate and throughput ride its entry, where the poll
    # used to answer the first rank alone.
    assert (loads[1].cache_hit_rate, loads[1].gen_throughput) == (1.0, 100.0)
    assert (loads[0].cache_hit_rate, loads[0].gen_throughput) == (0.0, 0.0)
    for load in loads:
        record = servicer._load_record(load.dp_rank)
        assert _core(record) == (load.num_running_reqs, load.num_waiting_reqs, load.token_usage)
        assert (record.cache_hit_rate, record.gen_throughput) == (
            load.cache_hit_rate,
            load.gen_throughput,
        )


def test_get_loads_lists_a_rank_without_a_snapshot_only_when_it_is_the_first():
    from types import SimpleNamespace

    # Only the second engine has stepped: the first rank is zero-filled
    # before its first snapshot, as the poll always answered.
    servicer = _servicer_over(
        SimpleNamespace(last_scheduler_stats_dict={1: _stats(5, 1, 0.5)}), managed=[0, 1]
    )
    assert [(load.dp_rank, load.num_running_reqs) for load in servicer._scheduler_loads()] == [
        (0, 0),
        (1, 5),
    ]
    # Only the first has: the other rank gets no entry rather than zeros
    # that read as idle.
    servicer = _servicer_over(
        SimpleNamespace(last_scheduler_stats_dict={0: _stats(3, 2, 0.25)}), managed=[0, 1]
    )
    assert [(load.dp_rank, load.num_running_reqs) for load in servicer._scheduler_loads()] == [
        (0, 3)
    ]
    # A servicer managing a slice of the ranks (external or hybrid load
    # balancing) lists its own ranks, in order.
    servicer = _servicer_over(
        SimpleNamespace(last_scheduler_stats_dict={5: _stats(5, 1, 0.5), 4: _stats(3, 2, 0.25)}),
        managed=[5, 4],
    )
    assert [(load.dp_rank, load.num_running_reqs) for load in servicer._scheduler_loads()] == [
        (4, 3),
        (5, 5),
    ]
    # No managed list (one engine): the first rank alone, zero-filled
    # before its first snapshot.
    servicer = _servicer_over(SimpleNamespace())
    assert [(load.dp_rank, load.num_running_reqs) for load in servicer._scheduler_loads()] == [
        (0, 0)
    ]


def test_the_first_rank_falls_back_to_the_configured_start_rank():
    from types import SimpleNamespace

    servicer = _servicer_over(SimpleNamespace(last_scheduler_stats_dict={3: _stats(1, 0, 0.1)}))
    servicer.engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=3)
    )
    assert servicer._first_rank() == 3
    record = servicer._load_record(None)
    assert _core(record) == (1, 0, 0.1) and record.HasField("waiting_uncached_tokens")
