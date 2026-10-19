"""Fast adapter regressions; native metric formulas belong to vLLM's tests."""

import asyncio
import enum
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import msgspec
import pytest
from smg_grpc_servicer.vllm.rust_ops import RustOps


@pytest.fixture
def ops(monkeypatch):
    """Stub only the native API boundary; exercise real adapter state and HTTP handling."""

    def module(name, **attrs):
        """Install isolated engine interfaces without importing a GPU runtime."""
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    class Reason(enum.IntEnum):
        STOP, LENGTH, ABORT, ERROR, REPETITION = range(5)

        def __str__(self):
            """Use the native wire spelling for terminal reasons."""
            return self.name.lower()

    @dataclass
    class Scheduler:
        num_running_reqs: int = 0

    module("vllm", envs=NS(VLLM_LOG_STATS_INTERVAL=0))
    module(
        "vllm.v1.engine",
        EngineCoreRequest=NS(),
        EngineCoreOutput=NS(),
        EngineCoreOutputs=NS(),
        FinishReason=Reason,
        EngineCoreEvent=msgspec.defstruct("Event", []),
    )
    module("vllm.v1.engine.core_client", MPClient=Mock())
    module("vllm.v1.engine.parallel_sampling", ParentRequest=Mock())
    module(
        "vllm.v1.metrics.stats",
        LoRARequestStates=Mock(),
        IterationStats=Mock(side_effect=Mock),
        SchedulerStats=Scheduler,
        PrefillStats=msgspec.defstruct("Prefill", []),
        RequestStateStats=lambda **kw: NS(num_generation_tokens=0, **kw),
    )
    module("vllm.v1.metrics.loggers", StatLoggerManager=Mock())
    from prometheus_client import CollectorRegistry

    module(
        "vllm.v1.metrics.prometheus",
        setup_multiprocess_prometheus=Mock(),
        shutdown_prometheus=Mock(),
        get_prometheus_registry=lambda: CollectorRegistry(),
    )
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    result = RustOps(
        NS(parallel_config=NS(data_parallel_rank=0, data_parallel_size=1)),
        0,
        log_stats=True,
        host="127.0.0.1",
    )
    result.observe("ready_done", b"")
    yield result
    asyncio.run(result.close())


def send(ops, event, **data):
    """Send a real msgpack metadata frame through the observer."""
    ops.observe(event, msgspec.msgpack.encode(data))
    assert ops.error is None


def add(ops, name):
    """Admit one request with a stable arrival time."""
    send(ops, "add", request_id=name, prompt_token_ids=[1], arrival_time=100.0)


def output(ops, name, **extra):
    """Deliver one token and a scheduler snapshot without consuming it."""
    send(
        ops,
        "output",
        outputs=[dict(request_id=name, new_token_ids=[7], **extra)],
        scheduler_stats={"num_running_reqs": 1},
        timestamp=12.0,
    )


def test_slow_stream_and_producer_timestamp(ops):
    """A delayed stream cannot block newer gauges or unrelated completed iterations."""
    for name in ("slow", "fast"):
        add(ops, name)
        output(ops, name)
    snapshots = ops.stats.record.call_args_list
    assert len(snapshots) == 2
    assert all(call.kwargs["scheduler_stats"].num_running_reqs == 1 for call in snapshots)
    ops.observe("consume", b"fast", 101.0)
    ops.observe("processed", b'["fast", 0]')
    iteration = ops.stats.record.call_args.kwargs["iteration_stats"]
    assert iteration.iteration_timestamp == 101.0
    iteration.update_from_output.assert_called_once()
    assert ops.pending_outputs == 1
    ops.observe("abort", b"slow")
    assert ops.error is None and ops.pending_outputs == 0


@pytest.mark.parametrize(
    "reason,frontend_stop,expected", [(4, False, 4), (4, True, 0), (1, False, 1)]
)
def test_effective_finish_reason_and_cleanup(ops, reason, frontend_stop, expected):
    """Preserve engine repetition unless a frontend string stop wins, then free state."""
    add(ops, "r")
    output(ops, "r", finish_reason=reason)
    ops.observe("consume", b"r", 101.0)
    iteration = ops.requests["r"].active.iteration
    ops.observe(
        "finish", msgspec.json.encode(["r", 0, "length" if reason == 1 else "stop", frontend_stop])
    )
    assert iteration.update_from_finished_request.call_args.kwargs["finish_reason"] == expected
    assert ops.error is None and not ops.requests and ops.pending_outputs == 0


@pytest.mark.parametrize("failure", ["overflow", "invalid", "backlog"])
def test_collection_failure_releases_state(ops, monkeypatch, failure):
    """Loss or invalid metadata must invalidate collection instead of retaining a backlog."""
    add(ops, "r")
    if failure == "backlog":
        monkeypatch.setattr("smg_grpc_servicer.vllm.rust_ops.MAX_PENDING_OUTPUTS", 0)
        ops.observe(
            "output",
            msgspec.msgpack.encode(
                {"outputs": [{"request_id": "r", "new_token_ids": [7]}], "timestamp": 12}
            ),
        )
    else:
        ops.observe("overflow" if failure == "overflow" else "output", b"invalid")
    assert ops.error is not None and not ops.requests and ops.pending_outputs == 0
    ops.observe("add", b"ignored")
    assert not ops.requests


def test_parallel_children_keep_arrival_and_cancel_together(ops):
    """Preserve frontend arrival times and release all children when a parent is dropped."""
    ops.observe("parent", b'["r", 2, 95.0]')
    for child in ("r-0", "r-1"):
        add(ops, child)
        assert ops.requests[child].stats.arrival_time == 95.0
    ops.observe("forget_parent", b"r")
    assert ops.error is None
    assert not (ops.requests or ops.parents or ops.children or ops.arrivals)


def test_incompatible_api_fails_at_startup(ops, monkeypatch):
    """Reject native signature drift before setting up multiprocess metrics."""
    stats = sys.modules["vllm.v1.metrics.stats"]
    monkeypatch.setattr(stats.IterationStats, "update_from_output", lambda self: None)
    with pytest.raises(RuntimeError, match="incompatible with this vLLM"):
        RustOps(ops.config, 0, log_stats=True)
    assert "PROMETHEUS_MULTIPROC_DIR" not in os.environ


def test_logging_failure_recovers(ops):
    """A text logger failure must not disable otherwise valid collection."""

    async def exercise():
        """Wait for a successful logging iteration following one injected failure."""
        done = asyncio.Event()
        calls = 0

        def log():
            """Fail once, then signal a successful retry."""
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary")
            done.set()

        ops.stats.log.side_effect = log
        ops.stats_task = asyncio.create_task(ops._log_stats())
        try:
            await asyncio.wait_for(done.wait(), 2)
            assert ops.error is None
        finally:
            await ops.close()

    asyncio.run(exercise())


def test_http_listener_health_host_and_cleanup(ops, monkeypatch):
    """Serve the configured interface and return 503 after losing metrics events."""
    import aiohttp
    from aiohttp import web

    sites = []
    site_type = web.TCPSite

    def bind(runner, host, port):
        """Capture the ephemeral listener to verify its actual bound interface."""
        assert host == "127.0.0.1"
        site = site_type(runner, host, port)
        sites.append(site)
        return site

    monkeypatch.setattr(web, "TCPSite", bind)

    async def exercise():
        """Scrape a real loopback socket without a model or vLLM checkout."""
        server = NS(set_metrics_observer=Mock())
        await ops.start(server)
        server.set_metrics_observer.assert_called_once_with(ops.observe)
        port = sites[0]._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                for expected in (200, 503):
                    async with session.get(f"http://127.0.0.1:{port}/metrics") as response:
                        assert response.status == expected
                    ops.observe("overflow", b"")
        finally:
            await ops.close()
            await ops.close()
        assert not sites[0]._server.is_serving()

    asyncio.run(exercise())


def test_disabled_stats_and_failed_bind_cleanup(ops):
    """Disabled collection stays inert and startup failures can be cleaned up twice."""
    import socket

    async def exercise():
        """Reserve a port to force a real listener startup error."""
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            ops.port = sock.getsockname()[1]
            ops.log_stats = False
            ops.observe("output", b"invalid")
            assert ops.error is None
            with pytest.raises(OSError):
                await ops.start(NS(set_metrics_observer=Mock()))
            await ops.close()
            await ops.close()

    asyncio.run(exercise())
