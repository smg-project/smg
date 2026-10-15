"""The TokenSpeed Python servicer's ``GetServerInfo`` carries the engine's
reasoning parser and where it starts a request's grammar on a thinking prompt
as ``server_args`` keys, the Router's label source for this engine."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2  # noqa: E402

_ROOT = Path(__file__).parents[1] / "smg_grpc_servicer" / "tokenspeed"


@pytest.fixture
def servicer_mod(monkeypatch):
    # Importing the engine (and torch) loads the runtime's dependencies; these
    # stand-ins cover the servicer module's imports and nothing of
    # GetServerInfo's body.
    symbols = {
        "torch": "",
        "tokenspeed.runtime.multimodal.inputs": "Modality MultimodalDataItem MultimodalInputs",
        "tokenspeed.runtime.multimodal.shm_transport": "ShmTensorHandle",
        "tokenspeed.runtime.pd.kv_events": "KVEventBatch",
    }
    for name, attrs in symbols.items():
        parts = name.split(".")
        for end in range(1, len(parts)):
            parent = ".".join(parts[:end])
            if parent not in sys.modules:
                module = ModuleType(parent)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, parent, module)
        module = ModuleType(name)
        for attr in attrs.split():
            setattr(module, attr, object)
        monkeypatch.setitem(sys.modules, name, module)
    name = "smg_grpc_servicer.tokenspeed.servicer"
    spec = importlib.util.spec_from_file_location(name, _ROOT / "servicer.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _server_info(servicer_mod, monkeypatch, **server_args):
    monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)
    monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    cls = servicer_mod.TokenSpeedSchedulerServicer
    servicer = cls.__new__(cls)
    servicer.server_args = SimpleNamespace(
        model="m", mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1)), **server_args
    )
    servicer.scheduler_info = {}
    servicer.start_time = 0.0
    servicer.async_llm = SimpleNamespace(rid_to_state={})
    return asyncio.run(
        servicer.GetServerInfo(tokenspeed_scheduler_pb2.GetServerInfoRequest(), None)
    )


def test_server_info_carries_where_the_engine_starts_a_grammar(servicer_mod, monkeypatch):
    # This engine applies a grammar from the first generated token, with or
    # without a reasoning parser, so the Router sends the thought and then the
    # calls; the parser rides along under its own name.
    gated = _server_info(servicer_mod, monkeypatch, reasoning_parser="kimi_k25").server_args
    assert gated["constrained_decoding_start"] == "first_token"
    assert gated["engine_reasoning_parser"] == "kimi_k25"
    # The engine's own argument keeps its name; it is not the Router's parser
    # override, which the Router reads under `reasoning_parser` from labels.
    assert gated["reasoning_parser"] == "kimi_k25"

    plain = _server_info(servicer_mod, monkeypatch, reasoning_parser=None).server_args
    assert plain["constrained_decoding_start"] == "first_token"
    assert "engine_reasoning_parser" not in plain
    assert plain["model"] == "m", "the other server args are reported as before"


def test_server_info_reports_the_engines_constrained_decoding_policy(servicer_mod, monkeypatch):
    """The engine's declared policy (a grammar on every turn) reaches the
    gateway as a field of its own, spelled as the `constrained_decoding` label
    reads it; an engine that declares none leaves the field at its default."""
    always = _server_info(servicer_mod, monkeypatch, constrained_decoding="ALWAYS")
    assert always.constrained_decoding == "always"
    requested = _server_info(servicer_mod, monkeypatch, constrained_decoding="requested")
    assert requested.constrained_decoding == "requested"
    silent = _server_info(servicer_mod, monkeypatch)
    assert silent.constrained_decoding == ""
    unknown = _server_info(servicer_mod, monkeypatch, constrained_decoding="sometimes")
    assert unknown.constrained_decoding == ""
