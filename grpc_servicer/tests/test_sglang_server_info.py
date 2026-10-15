"""The SGLang Python servicer's ``GetServerInfo`` carries the engine's reasoning
parser and where it starts a request's grammar on a thinking prompt as
``server_args`` keys, the Router's label source for this engine."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

msgspec = pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import sglang_scheduler_pb2  # noqa: E402

_ROOT = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang"


def _load(monkeypatch, name, filename):
    spec = importlib.util.spec_from_file_location(name, _ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def servicer_mod(monkeypatch):
    # Importing the engine loads its runtime's dependencies; these stand-ins
    # cover the servicer module's imports and nothing of GetServerInfo's body.
    symbols = {
        "sglang.srt.disaggregation.utils": "DisaggregationMode FAKE_BOOTSTRAP_HOST",
        "sglang.srt.managers.io_struct": (
            "AbortReq BatchEmbeddingOutput BatchTokenIDOutput FlushCacheReqOutput "
            "GetInternalStateReqOutput HealthCheckOutput ProfileReqOutput "
            "TokenizedEmbeddingReqInput TokenizedGenerateReqInput FlushCacheReqInput "
            "ProfileReq ProfileReqType"
        ),
        "sglang.srt.managers.load_snapshot": "LoadSnapshot create_load_snapshot_reader",
        "sglang.srt.observability.req_time_stats": (
            "APIServerReqTimeStats calibrate_time_diff real_time"
        ),
        "sglang.srt.server_args": "PortArgs ServerArgs",
        "sglang.srt.utils": "get_or_create_event_loop kill_process_tree get_bool_env_var",
        "sglang.srt.utils.network": "get_zmq_socket",
        "sglang.utils": "get_exception_traceback",
        "sglang.srt.configs.model_config": "ModelConfig",
        "sglang.srt.disaggregation.kv_events": "KVEventsConfig",
        "sglang.srt.managers.schedule_batch": (
            "Modality MultimodalDataItem MultimodalProcessorOutput"
        ),
        "sglang.srt.sampling.sampling_params": "SamplingParams",
        "sglang.srt.utils.hf_transformers_utils": "get_tokenizer",
        "smg_grpc_servicer.tensor_wire": "tensor_from_parts",
    }
    for name, attrs in symbols.items():
        parts = name.split(".")
        for end in range(1, len(parts)):
            parent = ".".join(parts[:end])
            if parent.startswith("sglang") and parent not in sys.modules:
                module = ModuleType(parent)
                module.__path__ = []
                module.__version__ = "0.0.0"
                monkeypatch.setitem(sys.modules, parent, module)
        module = ModuleType(name)
        for attr in attrs.split():
            setattr(module, attr, SimpleNamespace if attr == "AbortReq" else object)
        monkeypatch.setitem(sys.modules, name, module)
    _load(monkeypatch, "smg_grpc_servicer.sglang.request_manager", "request_manager.py")
    return _load(monkeypatch, "smg_grpc_servicer.sglang.servicer", "servicer.py")


class _ServerArgs(msgspec.Struct):
    """The engine's server arguments as the servicer holds them (a msgspec
    Struct, which `GetServerInfo` converts with msgspec's own asdict)."""

    model_path: str = "org/model"
    reasoning_parser: str | None = None


class _OlderServerArgs(msgspec.Struct):
    """Server arguments from before the engine had a reasoning parser."""

    model_path: str = "org/model"


class _WindowServerArgs(msgspec.Struct):
    """Server arguments with the running-window flag, None unless set."""

    model_path: str = "org/model"
    max_running_requests: int | None = None
    dp_size: int = 1
    enable_dp_attention: bool = False
    attn_dp_size: int | None = None


def _servicer(servicer_mod, server_args, **request_manager):
    servicer = servicer_mod.SGLangSchedulerServicer.__new__(servicer_mod.SGLangSchedulerServicer)
    servicer.server_args = server_args
    servicer.scheduler_info = {}
    servicer.start_time = 0.0
    servicer.request_manager = SimpleNamespace(
        get_server_info=lambda: {"active_requests": 0, "paused": False, "last_receive_time": 0.0},
        **request_manager,
    )
    return servicer


def _info(servicer):
    return asyncio.run(servicer.GetServerInfo(sglang_scheduler_pb2.GetServerInfoRequest(), None))


def _server_info(servicer_mod, server_args):
    return _info(_servicer(servicer_mod, server_args))


def _snapshots(*windows):
    return [
        SimpleNamespace(dp_rank=rank, max_running_requests=window)
        for rank, window in enumerate(windows)
    ]


def test_server_info_reports_the_window_the_scheduler_computed_when_the_flag_is_unset(
    servicer_mod, monkeypatch
):
    """The gateway reads the worker's running window (its PD admission bound) off
    `max_running_requests`; unset, the scheduler computes one and publishes it
    with its load snapshots, and that figure is what the servicer reports."""
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    reader = SimpleNamespace(read_all=lambda: _snapshots(1574))
    computed = _servicer(servicer_mod, _WindowServerArgs(), load_snapshot_reader=reader)
    assert _info(computed).server_args["max_running_requests"] == 1574

    # The operator's flag stays the figure reported, whatever the scheduler publishes.
    explicit = _servicer(
        servicer_mod, _WindowServerArgs(max_running_requests=8), load_snapshot_reader=reader
    )
    assert _info(explicit).server_args["max_running_requests"] == 8


def test_server_info_reports_the_window_in_the_flags_units_under_attention_dp(
    servicer_mod, monkeypatch
):
    """Under --enable-dp-attention the engine splits the flag across its
    attention-DP workers and each publishes its share; the figure reported is
    the whole worker's, as the flag would be. Plain data parallelism runs the
    whole flag on every scheduler and reports the published figure as it is."""
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    reader = SimpleNamespace(read_all=lambda: _snapshots(*([2048] * 8)))
    attention = _servicer(
        servicer_mod,
        _WindowServerArgs(dp_size=8, enable_dp_attention=True),
        load_snapshot_reader=reader,
    )
    assert _info(attention).server_args["max_running_requests"] == 16384

    replicas = _servicer(servicer_mod, _WindowServerArgs(dp_size=8), load_snapshot_reader=reader)
    assert _info(replicas).server_args["max_running_requests"] == 2048

    # The operator's flag is already in its own units.
    explicit = _servicer(
        servicer_mod,
        _WindowServerArgs(max_running_requests=16384, dp_size=8, enable_dp_attention=True),
        load_snapshot_reader=reader,
    )
    assert _info(explicit).server_args["max_running_requests"] == 16384


def test_a_partial_read_is_reported_but_not_cached_until_every_rank_published(
    servicer_mod, monkeypatch
):
    """The reader skips the ranks that have not published yet; a read with one
    of two ranks is reported as it stands, and the rank that publishes a
    smaller bound later still corrects it. Once both have published, the
    figure is cached."""
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    published = _snapshots(2048)
    reader = SimpleNamespace(read_all=lambda: list(published))
    servicer = _servicer(servicer_mod, _WindowServerArgs(dp_size=2), load_snapshot_reader=reader)
    assert _info(servicer).server_args["max_running_requests"] == 2048
    published.append(SimpleNamespace(dp_rank=1, max_running_requests=1024))
    assert _info(servicer).server_args["max_running_requests"] == 1024
    # Both ranks published: cached from here on.
    published.clear()
    assert _info(servicer).server_args["max_running_requests"] == 1024


def test_the_window_follows_the_rewritten_arguments_of_a_newer_engine(servicer_mod, monkeypatch):
    """A newer engine resolves --dp-size 8 --enable-dp-attention into
    attn_dp_size=8, dp_size=1, enable_dp_attention=False and runs one scheduler
    per attention-DP worker: the flag's units and the rank count follow, so a
    default start reports the whole engine's window (not one worker's share)
    and caches it once all eight published, not after the first."""
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    published = _snapshots(*([2048] * 7))
    reader = SimpleNamespace(read_all=lambda: list(published))
    servicer = _servicer(
        servicer_mod,
        _WindowServerArgs(attn_dp_size=8, dp_size=1, enable_dp_attention=False),
        load_snapshot_reader=reader,
    )
    # Seven of eight ranks published: reported in the flag's units, not cached.
    assert _info(servicer).server_args["max_running_requests"] == 16384
    published.append(SimpleNamespace(dp_rank=7, max_running_requests=1024))
    assert _info(servicer).server_args["max_running_requests"] == 8192
    published.clear()
    assert _info(servicer).server_args["max_running_requests"] == 8192

    # Two replicas of four attention-DP workers each: the flag is split four
    # ways, eight schedulers publish.
    published = _snapshots(*([1024] * 7))
    mixed = _servicer(
        servicer_mod,
        _WindowServerArgs(attn_dp_size=4, dp_size=2, enable_dp_attention=False),
        load_snapshot_reader=SimpleNamespace(read_all=lambda: list(published)),
    )
    assert _info(mixed).server_args["max_running_requests"] == 4096
    published.append(SimpleNamespace(dp_rank=7, max_running_requests=512))
    assert _info(mixed).server_args["max_running_requests"] == 2048
    published.clear()
    assert _info(mixed).server_args["max_running_requests"] == 2048


def test_server_info_reports_the_window_once_a_snapshot_is_published(servicer_mod, monkeypatch):
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    published = []
    reader = SimpleNamespace(read_all=lambda: list(published))
    servicer = _servicer(servicer_mod, _WindowServerArgs(), load_snapshot_reader=reader)
    # Before the schedulers' first snapshot the argument is reported as it is.
    assert _info(servicer).server_args["max_running_requests"] is None
    published.extend(_snapshots(0, 314))
    assert _info(servicer).server_args["max_running_requests"] == 314
    # Cached from then on: the bound does not change for the engine's lifetime.
    published.clear()
    assert _info(servicer).server_args["max_running_requests"] == 314

    # A request manager without the reader (an older layout) reports as before.
    bare = _server_info(servicer_mod, _WindowServerArgs())
    assert bare.server_args["max_running_requests"] is None


def test_server_info_carries_where_the_engine_starts_a_grammar(servicer_mod, monkeypatch):
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    gated = _server_info(servicer_mod, _ServerArgs(reasoning_parser="qwen3")).server_args
    # Behind --reasoning-parser the engine holds a request's grammar back until
    # the thought's end, so the Router sends the bare calls.
    assert gated["constrained_decoding_start"] == "after_reasoning"
    assert gated["engine_reasoning_parser"] == "qwen3"
    # The engine's own argument keeps its name; it is not the Router's parser
    # override, which the Router reads under `reasoning_parser` from labels.
    assert gated["reasoning_parser"] == "qwen3"

    plain = _server_info(servicer_mod, _ServerArgs(reasoning_parser=None)).server_args
    assert plain["constrained_decoding_start"] == "first_token"
    assert "engine_reasoning_parser" not in plain

    older = _server_info(servicer_mod, _OlderServerArgs()).server_args
    assert "constrained_decoding_start" not in older and "engine_reasoning_parser" not in older
    assert older["model_path"] == "org/model", "the other server args are reported as before"
