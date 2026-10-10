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


def _server_info(servicer_mod, server_args):
    servicer = servicer_mod.SGLangSchedulerServicer.__new__(servicer_mod.SGLangSchedulerServicer)
    servicer.server_args = server_args
    servicer.scheduler_info = {}
    servicer.start_time = 0.0
    servicer.request_manager = SimpleNamespace(
        get_server_info=lambda: {"active_requests": 0, "paused": False, "last_receive_time": 0.0}
    )
    return asyncio.run(servicer.GetServerInfo(sglang_scheduler_pb2.GetServerInfoRequest(), None))


def test_server_info_carries_where_the_engine_starts_a_grammar(servicer_mod, monkeypatch):
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    gated = _server_info(servicer_mod, _ServerArgs(reasoning_parser="qwen3")).server_args
    # Behind --reasoning-parser the engine holds a request's grammar back until
    # the thought's end, so the Router sends the bare calls.
    assert gated["structured_outputs_start"] == "after_reasoning"
    assert gated["engine_reasoning_parser"] == "qwen3"
    # The engine's own argument keeps its name; it is not the Router's parser
    # override, which the Router reads under `reasoning_parser` from labels.
    assert gated["reasoning_parser"] == "qwen3"

    plain = _server_info(servicer_mod, _ServerArgs(reasoning_parser=None)).server_args
    assert plain["structured_outputs_start"] == "first_token"
    assert "engine_reasoning_parser" not in plain

    older = _server_info(servicer_mod, _OlderServerArgs()).server_args
    assert "structured_outputs_start" not in older and "engine_reasoning_parser" not in older
    assert older["model_path"] == "org/model", "the other server args are reported as before"
