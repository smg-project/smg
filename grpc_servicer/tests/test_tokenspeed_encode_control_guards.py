"""An encode worker's discovery servicer refuses the scheduler's control RPCs
by name instead of forwarding them to the encode loop.

The TokenSpeed Python servicer mounts ``TokenSpeedEncodeDiscoveryServicer`` on
an encode worker so the gateway's worker discovery (health, model and server
info over the scheduler stub) can register it. The class is the full scheduler
servicer with ``Generate`` refused by name; ``FlushCache``, ``StartProfile``
and ``StopProfile`` were inherited as implemented and forwarded to the
worker, whose encode loop has no control-message dispatch (a forwarded control
input is taken as an encode request). They are refused UNIMPLEMENTED, by RPC
name, before anything reaches the worker.

Run with: pytest grpc_servicer/tests/test_tokenspeed_encode_control_guards.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import grpc
import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import common_pb2, tokenspeed_scheduler_pb2  # noqa: E402

_SERVER = Path(__file__).parents[1] / "smg_grpc_servicer" / "tokenspeed" / "server.py"


class _Aborted(Exception):
    """Raised by the fake context's ``abort``, as grpc.aio's never returns."""


class _Context:
    """A servicer context that records its one abort."""

    def __init__(self):
        self.aborts = []

    async def abort(self, code, details="", trailing_metadata=()):
        self.aborts.append((code, details))
        raise _Aborted()


class _Worker:
    """The worker's ``AsyncLLM`` as the servicer holds it: a control method
    that is reached records itself and fails, the way an encode loop fails on
    a control input it cannot dispatch."""

    def __init__(self):
        self.reached = []

    async def flush_cache(self):
        self.reached.append("flush_cache")
        raise RuntimeError("flush_cache reached the encode worker")

    async def start_profile(self, **_kwargs):
        self.reached.append("start_profile")
        raise RuntimeError("start_profile reached the encode worker")

    async def stop_profile(self):
        self.reached.append("stop_profile")
        raise RuntimeError("stop_profile reached the encode worker")


def _install(monkeypatch, name, **attrs):
    """A stand-in module under ``name``, its missing parents as empty packages
    (the engine's own package tree is stubbed the same way by
    ``tokenspeed_servicer_mod``)."""
    parts = name.split(".")
    for end in range(1, len(parts)):
        parent = ".".join(parts[:end])
        if parent not in sys.modules:
            package = ModuleType(parent)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, parent, package)
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def server_mod(monkeypatch, tokenspeed_servicer_mod):
    """The TokenSpeed server module over the real scheduler servicer
    (``tokenspeed_servicer_mod``, the engine's imports stubbed), with stand-ins
    for the two imports that load the engine at module level: the server args
    type and the scheduler launcher."""
    _install(
        monkeypatch, "tokenspeed.runtime.utils.server_args", ServerArgs=type("ServerArgs", (), {})
    )
    _install(
        monkeypatch,
        "smg_grpc_servicer.tokenspeed.scheduler_launcher",
        launch_engine=lambda _args: (object(), {}),
    )
    spec = importlib.util.spec_from_file_location("test_tokenspeed_encode_server", _SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discovery_servicer(server_mod):
    """The encode worker's discovery servicer as ``serve_grpc`` builds it,
    without the engine: the worker fake, the encode-role args."""
    cls = server_mod.TokenSpeedEncodeDiscoveryServicer
    servicer = cls.__new__(cls)
    servicer.async_llm = _Worker()
    servicer.server_args = SimpleNamespace(disaggregation_mode="encode")
    servicer.scheduler_info = {}
    return servicer


_CONTROL_RPCS = [
    ("FlushCache", lambda: common_pb2.FlushCacheRequest(timeout_s=0.0)),
    ("StartProfile", lambda: common_pb2.StartProfileRequest()),
    ("StopProfile", lambda: common_pb2.StopProfileRequest()),
]


@pytest.mark.parametrize(
    ("rpc", "request_of"), _CONTROL_RPCS, ids=[rpc for rpc, _ in _CONTROL_RPCS]
)
def test_a_control_rpc_is_refused_by_name_before_the_worker(server_mod, rpc, request_of):
    servicer = _discovery_servicer(server_mod)
    context = _Context()

    with pytest.raises(_Aborted):
        asyncio.run(getattr(servicer, rpc)(request_of(), context))

    assert context.aborts == [
        (grpc.StatusCode.UNIMPLEMENTED, f"{rpc} is unavailable on TokenSpeed encode workers")
    ]
    assert servicer.async_llm.reached == []


def test_generate_is_refused_by_name_as_before(server_mod):
    servicer = _discovery_servicer(server_mod)
    context = _Context()

    async def first_item():
        stream = servicer.Generate(tokenspeed_scheduler_pb2.GenerateRequest(), context)
        return await stream.__anext__()

    with pytest.raises(_Aborted):
        asyncio.run(first_item())

    assert context.aborts == [
        (
            grpc.StatusCode.FAILED_PRECONDITION,
            "Generate is unavailable on TokenSpeed encode workers",
        )
    ]


def test_the_scheduler_servicer_itself_still_forwards_a_control_rpc(tokenspeed_servicer_mod):
    """The refusal belongs to the encode role only: the scheduler servicer the
    discovery surface inherits from forwards ``StopProfile`` to its worker (a
    prefill or decode loop dispatches it) and reports the worker's failure."""
    cls = tokenspeed_servicer_mod.TokenSpeedSchedulerServicer
    servicer = cls.__new__(cls)
    servicer.async_llm = _Worker()
    context = _Context()

    with pytest.raises(_Aborted):
        asyncio.run(servicer.StopProfile(common_pb2.StopProfileRequest(), context))

    assert servicer.async_llm.reached == ["stop_profile"]
    assert context.aborts[0][0] is grpc.StatusCode.INTERNAL
