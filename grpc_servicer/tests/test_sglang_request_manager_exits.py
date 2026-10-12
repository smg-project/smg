"""The SGLang Python servicer's request manager finishes a request's time
stats, and with them the request's root trace span, on every exit of the
request handler: the terminal output, a client cancel mid-stream, the
scheduler's own abort of the request, a failed dispatch.

The engine's stats open the root span when the request is created and end it
when they are finished; the stand-in below does the same against an in-memory
exporter of ended spans, so a span the handler leaves open is a span the
exporter never receives (as a collector never would)."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("zmq")

_ROOT = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang"


class InMemorySpanExporter:
    """The spans that were ended, in order; an open span never arrives."""

    def __init__(self) -> None:
        self.spans: list[_Span] = []

    def names(self) -> list[str]:
        return [span.name for span in self.spans]


class _Span:
    def __init__(self, name: str, exporter: InMemorySpanExporter) -> None:
        self.name = name
        self.ended = 0
        self._exporter = exporter

    def end(self) -> None:
        self.ended += 1
        self._exporter.spans.append(self)


def _time_stats_class(exporter: InMemorySpanExporter):
    class APIServerReqTimeStats:
        """The engine's frontend-role time stats with tracing on: the created
        time opens the request's root span, the finished time ends it."""

        def __init__(self, **_kwargs) -> None:
            self.finished_time = 0.0
            self.root_span: _Span | None = None

        def set_created_time(self, ts=None) -> None:
            self.root_span = _Span(" Req", exporter)

        def set_finished_time(self, ts=None, span_attrs=None) -> None:
            self.finished_time = time.perf_counter()
            assert self.root_span is not None
            self.root_span.end()

        def set_api_server_dispatch_time(self, ts=None) -> None:
            pass

        def set_api_server_dispatch_finish_time(self, ts=None) -> None:
            pass

    return APIServerReqTimeStats


@pytest.fixture
def manager(monkeypatch):
    """A request manager over stand-ins for the engine's modules, with the
    span exporter its stats end into and the list of what it sent the
    scheduler."""
    exporter = InMemorySpanExporter()
    symbols = {
        "sglang.srt.disaggregation.utils": "DisaggregationMode",
        "sglang.srt.managers.io_struct": (
            "AbortReq BatchEmbeddingOutput BatchTokenIDOutput FlushCacheReqOutput "
            "GetInternalStateReqOutput HealthCheckOutput ProfileReqOutput "
            "TokenizedEmbeddingReqInput TokenizedGenerateReqInput"
        ),
        "sglang.srt.managers.load_snapshot": "LoadSnapshot create_load_snapshot_reader",
        "sglang.srt.observability.req_time_stats": (
            "APIServerReqTimeStats calibrate_time_diff real_time"
        ),
        "sglang.srt.server_args": "PortArgs ServerArgs",
        "sglang.srt.utils": "get_or_create_event_loop kill_process_tree",
        "sglang.srt.utils.network": "get_zmq_socket",
        "sglang.utils": "get_exception_traceback",
    }
    stand_ins = {
        "AbortReq": SimpleNamespace,
        "APIServerReqTimeStats": _time_stats_class(exporter),
        "calibrate_time_diff": lambda: None,
        "real_time": time.time,
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
            setattr(module, attr, stand_ins.get(attr, object))
        monkeypatch.setitem(sys.modules, name, module)
    name = "smg_grpc_servicer.sglang.request_manager"
    spec = importlib.util.spec_from_file_location(name, _ROOT / "request_manager.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    sent: list = []
    rm = module.GrpcRequestManager.__new__(module.GrpcRequestManager)
    rm.rid_to_state = {}
    rm.disaggregation_mode = "null"
    rm.crash_dump_request_list = []
    rm.enable_trace = False  # the stand-in stats start the trace themselves
    rm.send_to_scheduler = SimpleNamespace(send_pyobj=sent.append)
    rm.sent = sent
    rm.exporter = exporter
    return rm


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def _handle(rm, rid: str):
    """Run the request handler the way ``Generate`` consumes it, until it
    waits for the scheduler's first output."""
    got: list = []

    async def consume():
        async for item in rm._handle_single_request(SimpleNamespace(stream=True), rid, None):
            got.append(item)

    task = asyncio.create_task(consume())
    await _settle()
    assert rid in rm.rid_to_state and rm.sent[-1].rid == rid
    return task, got


def _finished(rm, rid: str) -> None:
    """What the output path does on the request's terminal output."""
    state = rm.rid_to_state[rid]
    state.time_stats.set_finished_time()
    state.out_queue.put_nowait({"request_id": rid, "token_ids": [2], "finished": True})


def test_terminal_output_ends_the_root_span_once(manager):
    async def scenario():
        task, got = await _handle(manager, "r-done")
        manager.rid_to_state["r-done"].out_queue.put_nowait({"token_ids": [1], "finished": False})
        await _settle()
        _finished(manager, "r-done")
        await task
        return got

    got = asyncio.run(scenario())
    assert [item["finished"] for item in got] == [False, True]
    assert manager.exporter.names() == [" Req"]
    assert manager.exporter.spans[0].ended == 1, "the handler's exit does not end it again"
    assert "r-done" not in manager.rid_to_state


def test_client_cancel_mid_stream_ends_the_root_span(manager):
    async def scenario():
        task, got = await _handle(manager, "r-cancel")
        manager.rid_to_state["r-cancel"].out_queue.put_nowait({"token_ids": [1], "finished": False})
        await _settle()
        assert len(got) == 1, "the stream is under way"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert manager.exporter.names() == [" Req"], "the cancelled request's root span is ended"
    assert manager.exporter.spans[0].ended == 1
    assert "r-cancel" not in manager.rid_to_state
    assert [m.rid for m in manager.sent] == ["r-cancel", "r-cancel"], (
        "the abort reached the scheduler"
    )


def test_scheduler_abort_ends_the_root_span(manager):
    async def scenario():
        task, got = await _handle(manager, "r-abort")
        await manager._handle_abort_req(
            SimpleNamespace(
                rid="r-abort", finished_reason={"type": "abort", "message": "queue full"}
            )
        )
        await task
        return got

    got = asyncio.run(scenario())
    assert [item["error"] for item in got] == ["queue full"]
    assert manager.exporter.names() == [" Req"]
    assert manager.exporter.spans[0].ended == 1
    assert "r-abort" not in manager.rid_to_state


def test_failed_dispatch_ends_the_root_span(manager):
    def refuse(_obj):
        raise RuntimeError("scheduler socket closed")

    manager.send_to_scheduler = SimpleNamespace(send_pyobj=refuse)

    async def scenario():
        async for _ in manager._handle_single_request(SimpleNamespace(stream=True), "r-fail", None):
            pass

    with pytest.raises(RuntimeError, match="socket closed"):
        asyncio.run(scenario())
    assert manager.exporter.names() == [" Req"]
    assert "r-fail" not in manager.rid_to_state


def test_failed_embedding_dispatch_ends_the_root_span(manager):
    def refuse(_obj):
        raise RuntimeError("scheduler socket closed")

    manager.send_to_scheduler = SimpleNamespace(send_pyobj=refuse)

    async def scenario():
        future = await manager.embedding_request(SimpleNamespace(), "e-fail")
        with pytest.raises(RuntimeError, match="socket closed"):
            await future

    asyncio.run(scenario())
    assert manager.exporter.names() == [" Req"]
    assert "e-fail" not in manager.rid_to_state


# ---------------------------------------------------------------------------
# n > 1: the multiplexed sub-requests on the consumer's exit
# ---------------------------------------------------------------------------
#
# The checks run inside the loop, right after the consumer left: at the end of
# ``asyncio.run`` the loop cancels every task still pending and closes every
# async generator still open, which would abort the sub-requests by itself and
# hide a request manager that leaves them running.


def _multi(stream: bool, n: int):
    return SimpleNamespace(stream=stream, sampling_params=SimpleNamespace(n=n, max_new_tokens=8))


def _chunk(rm, rid: str) -> dict:
    """A non-terminal output for the request, as the output path queues it."""
    item = {"request_id": rid, "token_ids": [1], "finished": False}
    rm.rid_to_state[rid].out_queue.put_nowait(item)
    return item


async def _sub_requests_dispatched(rm, base: str, n: int) -> None:
    """Phase 1 (the prefix request) answered; phase 2 (the n sub-requests)
    dispatched and waiting on the scheduler."""
    await _settle()
    assert [m.rid for m in rm.sent] == [f"{base}-prefix"]
    _finished(rm, f"{base}-prefix")
    await _settle()
    await _settle()
    assert [m.rid for m in rm.sent] == [f"{base}-prefix"] + [f"{base}-{i}" for i in range(n)]


def _aborts(rm, after: int) -> list[str]:
    return sorted(m.rid for m in rm.sent[after:])


def _all_ended(rm, spans: int) -> None:
    """Every request's root span ended, once; no request state left."""
    assert rm.exporter.names() == [" Req"] * spans
    assert [span.ended for span in rm.exporter.spans] == [1] * spans
    assert rm.rid_to_state == {}


def test_client_cancel_of_a_streaming_n2_request_aborts_both_sub_requests(manager):
    async def scenario():
        got: list = []

        async def consume():
            async for item in manager.generate_request(_multi(True, 2), "r-multi", None):
                got.append(item)

        task = asyncio.create_task(consume())
        await _sub_requests_dispatched(manager, "r-multi", 2)
        _chunk(manager, "r-multi-0")
        await _settle()
        assert got == [{"request_id": "r-multi-0", "token_ids": [1], "finished": False, "index": 0}]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _settle()
        assert _aborts(manager, 3) == ["r-multi-0", "r-multi-1"], "both sub-requests aborted"
        _all_ended(manager, 3)

    asyncio.run(scenario())


def test_a_closed_multiplexed_stream_aborts_a_sub_request_whose_output_was_not_read(manager):
    """Both sub-requests have an output queued; the consumer reads one and
    leaves. One handler is parked at its yield behind the consumer, the other
    at its yield with its output unread: neither waits on anything a cancel
    could reach, so the abort is sent on the stream's close (what the event
    loop does with a generator its consumer dropped)."""

    async def scenario():
        got: list = []
        stream = manager.generate_request(_multi(True, 2), "r-unread", None)

        async def consume():
            async for item in stream:
                got.append(item)
                break

        task = asyncio.create_task(consume())
        await _sub_requests_dispatched(manager, "r-unread", 2)
        _chunk(manager, "r-unread-0")
        _chunk(manager, "r-unread-1")
        await task
        assert len(got) == 1
        assert _aborts(manager, 3) == [], "nothing aborted while the stream is merely suspended"
        await stream.aclose()
        await _settle()
        assert _aborts(manager, 3) == ["r-unread-0", "r-unread-1"], "both sub-requests aborted"
        _all_ended(manager, 3)

    asyncio.run(scenario())


def test_a_failing_sub_request_handler_aborts_its_running_sibling(manager):
    """One sub-request's dispatch fails while the other waits on the scheduler:
    the failure leaves the multiplex loop through its ``finally``, which aborts
    the sibling still running (before, it kept generating)."""
    sent = manager.sent

    def dispatch(obj):
        if getattr(obj, "rid", None) == "r-fail-0":
            raise RuntimeError("scheduler socket closed")
        sent.append(obj)

    manager.send_to_scheduler = SimpleNamespace(send_pyobj=dispatch)

    async def scenario():
        got: list = []

        async def consume():
            async for item in manager.generate_request(_multi(True, 2), "r-fail", None):
                got.append(item)

        task = asyncio.create_task(consume())
        await _settle()
        assert [m.rid for m in manager.sent] == ["r-fail-prefix"]
        _finished(manager, "r-fail-prefix")
        with pytest.raises(RuntimeError, match="socket closed"):
            await task
        await _settle()
        assert got == []
        assert [m.rid for m in manager.sent][:2] == ["r-fail-prefix", "r-fail-1"], (
            "the sibling was dispatched; the failing one never reached the scheduler"
        )
        assert _aborts(manager, 2) == ["r-fail-1"], "the running sibling is aborted"
        _all_ended(manager, 3)

    asyncio.run(scenario())


def test_client_cancel_of_a_non_streaming_n3_request_aborts_only_the_running_sub_request(manager):
    """The non-streaming path runs the sub-requests one after another: a cancel
    reaches the one in flight through its own handler, and the ones not yet
    started were never dispatched."""

    async def scenario():
        got: list = []

        async def consume():
            async for item in manager.generate_request(_multi(False, 3), "r-batch", None):
                got.append(item)

        task = asyncio.create_task(consume())
        await _settle()
        assert [m.rid for m in manager.sent] == ["r-batch-prefix"]
        _finished(manager, "r-batch-prefix")
        await _settle()
        await _settle()
        assert [m.rid for m in manager.sent] == ["r-batch-prefix", "r-batch-0"]
        _finished(manager, "r-batch-0")
        await _settle()
        await _settle()
        assert [m.rid for m in manager.sent] == ["r-batch-prefix", "r-batch-0", "r-batch-1"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _settle()
        assert got == []
        assert _aborts(manager, 3) == ["r-batch-1"], "the sub-request in flight is aborted"
        assert "r-batch-2" not in [m.rid for m in manager.sent], "the third was never dispatched"
        _all_ended(manager, 3)

    asyncio.run(scenario())
