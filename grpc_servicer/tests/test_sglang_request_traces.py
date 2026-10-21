"""The request traces the SGLang plugin keeps on the Rust servicer's wire, and
the request metadata frame that carries the Router's context (engine-free:
SGLang's tracing pieces are stand-ins that record the calls the tokenizer
manager makes behind SGLang's own frontend)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import msgspec
import pytest
from smg_grpc_servicer.sglang import request_traces

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
HEADERS = {"traceparent": TRACEPARENT, "tracestate": "vendor=a"}


def _install(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent:
        parent_module = sys.modules.get(parent) or _install(monkeypatch, parent)
        setattr(parent_module, child, module)
    return module


class FakeTraceContext:
    """SGLang's `TraceReqContext` as the stats see it: on or off."""

    def __init__(self, enabled: bool):
        self.tracing_enable = enabled


class FakeTimeStats:
    """SGLang's `APIServerReqTimeStats` as the tokenizer manager drives it;
    picklable (module level), so the scheduler's copy can be made."""

    tracing_on = True
    created: list[FakeTimeStats] = []

    def __init__(self, disagg_mode):
        self.disagg_mode = disagg_mode
        self.calls: list[tuple] = []
        self.trace_ctx = FakeTraceContext(False)
        FakeTimeStats.created.append(self)

    def init_trace_ctx(self, rid, bootstrap_room, external_trace_header=None):
        self.calls.append(("init_trace_ctx", rid, bootstrap_room, external_trace_header))
        self.trace_ctx = FakeTraceContext(FakeTimeStats.tracing_on)

    def set_created_time(self, ts=None):
        self.calls.append(("set_created_time",))

    def set_tokenize_finish_time(self, ts=None):
        self.calls.append(("set_tokenize_finish_time",))

    def set_finished_time(self, ts=None, span_attrs=None):
        self.calls.append(("set_finished_time",))


@pytest.fixture
def sglang_tracing(monkeypatch):
    """The SGLang pieces `RequestTraces` imports, as stand-ins."""
    monkeypatch.setattr(FakeTimeStats, "created", [])
    monkeypatch.setattr(FakeTimeStats, "tracing_on", True)
    switch = {"enabled": True}
    _install(
        monkeypatch,
        "sglang.srt.observability.trace",
        get_global_tracing_enabled=lambda: switch["enabled"],
    )
    _install(
        monkeypatch, "sglang.srt.observability.req_time_stats", APIServerReqTimeStats=FakeTimeStats
    )
    _install(monkeypatch, "sglang.srt.disaggregation.utils", DisaggregationMode=lambda mode: mode)
    _install(
        monkeypatch,
        "sglang.srt.runtime_context",
        get_disagg=lambda: SimpleNamespace(disaggregation_mode="null"),
    )
    return switch


def _request(rid="r1", bootstrap_room=None):
    return SimpleNamespace(rid=rid, bootstrap_room=bootstrap_room, time_stats=None)


def test_attach_starts_the_trace_from_the_context_and_hands_the_scheduler_a_copy(sglang_tracing):
    """smg-lab#182: the request's trace starts from the Router's context the
    way the tokenizer manager starts it (`init_trace_ctx` with the external
    headers, the root span on `set_created_time`, the tokenize slice closed at
    once); the request carries a pickled copy of the stats for the scheduler,
    while the original stays in the registry until the request ends."""
    traces = request_traces.RequestTraces()
    request = _request(bootstrap_room=7)

    assert traces.attach(request, HEADERS) is True

    (stats,) = FakeTimeStats.created
    assert stats.disagg_mode == "null"
    assert stats.calls == [
        ("init_trace_ctx", "r1", 7, HEADERS),
        ("set_created_time",),
        ("set_tokenize_finish_time",),
    ]
    assert len(traces) == 1
    assert isinstance(request.time_stats, FakeTimeStats) and request.time_stats is not stats
    assert request.time_stats.calls == stats.calls
    # The terminal output ends the root span, once.
    assert traces.finish("r1") is True
    assert stats.calls[-1] == ("set_finished_time",)
    assert request.time_stats.calls[-1] != ("set_finished_time",)  # the scheduler's copy
    assert len(traces) == 0 and traces.finish("r1") is False


def test_nothing_is_traced_without_a_context_or_with_tracing_off(sglang_tracing):
    traces = request_traces.RequestTraces()
    assert traces.attach(_request(), None) is False
    assert traces.attach(_request(), {}) is False
    assert FakeTimeStats.created == []

    sglang_tracing["enabled"] = False
    assert request_traces.RequestTraces().attach(_request(), HEADERS) is False
    assert FakeTimeStats.created == []

    # Tracing initialized but this request's context off (a trace level of 0,
    # a module filtered out): no registry entry, nothing on the request.
    FakeTimeStats.tracing_on = False
    request = _request()
    assert traces.attach(request, HEADERS) is False
    assert request.time_stats is None and len(traces) == 0


def test_a_duplicate_request_id_is_not_traced_twice(sglang_tracing):
    traces = request_traces.RequestTraces()
    assert traces.attach(_request("dup"), HEADERS) is True
    assert traces.attach(_request("dup"), HEADERS) is False
    assert len(FakeTimeStats.created) == 1


def test_a_failing_tracer_never_costs_the_request(sglang_tracing, caplog, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("no tracer")

    monkeypatch.setattr(FakeTimeStats, "init_trace_ctx", broken)
    traces = request_traces.RequestTraces()
    request = _request()
    with caplog.at_level("ERROR", logger=request_traces.logger.name):
        assert traces.attach(request, HEADERS) is False
    assert request.time_stats is None and len(traces) == 0
    assert "could not start the trace of r1" in caplog.text


def test_finish_batch_ends_the_finished_requests_only(sglang_tracing):
    traces = request_traces.RequestTraces()
    for rid in ("a", "b", "c"):
        assert traces.attach(_request(rid), HEADERS)
    assert traces.finish_batch(["a", "b", "c"], [None, {"type": "length"}, "abort"]) == 2
    assert len(traces) == 1
    finished = {stats.calls[0][1]: stats.calls[-1] for stats in FakeTimeStats.created}
    assert finished == {
        "a": ("set_tokenize_finish_time",),
        "b": ("set_finished_time",),
        "c": ("set_finished_time",),
    }


def test_tracing_is_off_without_the_engines_switch(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang", None)  # no SGLang at all: import fails
    monkeypatch.setitem(sys.modules, "sglang.srt.observability.trace", None)
    assert request_traces.tracing_enabled() is False
    assert request_traces.RequestTraces().attach(_request(), HEADERS) is False


# ---------------------------------------------------------------------------
# The metadata frame on the wire
# ---------------------------------------------------------------------------


def test_the_metadata_frame_is_read_loosely():
    wire = pytest.importorskip("smg_grpc_servicer.sglang.zmq_msgpack")
    assert wire.request_meta(msgspec.msgpack.encode({"trace_headers": HEADERS})) == {
        "trace_headers": HEADERS
    }
    assert wire.request_meta(msgspec.msgpack.encode({})) == {}
    assert wire.request_meta(b"\xc1not-msgpack") == {}  # an unreadable frame counts as empty
    assert wire.request_meta(msgspec.msgpack.encode([1, 2])) == {}


def test_an_add_with_a_metadata_frame_starts_the_requests_trace(monkeypatch, sglang_tracing):
    """The receiver hands the ADD's metadata frame to the registry: the
    request goes to the scheduler with its copy of the stats; an ADD without
    the frame, or without a registry, is decoded as before."""
    wire = pytest.importorskip("smg_grpc_servicer.sglang.zmq_msgpack")
    _install(monkeypatch, "sglang.srt.managers.io_struct", AbortReq=object)
    decoded: list = []

    class Decoder:
        @staticmethod
        def decode(payload):
            request = _request(rid=msgspec.msgpack.decode(payload)[1])
            decoded.append(request)
            return request

    monkeypatch.setattr(wire, "_DEC_ADD", Decoder())
    traces = request_traces.RequestTraces()
    payload = msgspec.msgpack.encode(["TokenizedGenerateReqInput", "t1"])
    meta = msgspec.msgpack.encode({"trace_headers": HEADERS})

    (request,) = wire.decode_request_frames([wire.REQ_TYPE_ADD, payload, meta], traces)
    assert request is decoded[0] and isinstance(request.time_stats, FakeTimeStats)
    assert len(traces) == 1 and FakeTimeStats.created[0].calls[0] == (
        "init_trace_ctx",
        "t1",
        None,
        HEADERS,
    )

    (plain,) = wire.decode_request_frames([wire.REQ_TYPE_ADD, payload], traces)
    assert plain.time_stats is None
    (untracked,) = wire.decode_request_frames([wire.REQ_TYPE_ADD, payload, meta])
    assert untracked.time_stats is None and len(traces) == 1
