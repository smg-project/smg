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
