"""Unit tests for the TokenSpeed KV-events config resolver + shared conversion.

Engine-free: no TokenSpeed install required. Modules are loaded by file path so
the ``tokenspeed`` package ``__init__`` (which imports the engine-bound
servicer) is never triggered.

Run with: pytest grpc_servicer/tests/test_tokenspeed_kv_events.py -v
"""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("smg_grpc_proto")

_SERVICER_ROOT = Path(__file__).parents[1] / "smg_grpc_servicer"


def _load(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _SERVICER_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts_kv_events = _load("tokenspeed/kv_events.py", "ts_kv_events")
shared = _load("kv_events.py", "shared_kv_events")


def test_tokenspeed_servicer_imports_current_kv_events_path():
    source = (_SERVICER_ROOT / "tokenspeed" / "servicer.py").read_text()

    assert "from tokenspeed.runtime.pd.kv_events import KVEventBatch" in source
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
