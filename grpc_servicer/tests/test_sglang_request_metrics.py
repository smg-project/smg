"""CPU-only regression tests for the gRPC request-metrics hooks.

Only SGLang engine imports are replaced; request state and lifecycle methods
are loaded from the production module. No scheduler or GPU is required.
"""

import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("grpc")
pytest.importorskip("zmq")


@pytest.fixture
def manager_mod(monkeypatch):
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
            setattr(module, attr, SimpleNamespace if attr == "AbortReq" else object)
        if "DisaggregationMode" in attrs.split():
            module.DisaggregationMode = SimpleNamespace(PREFILL="prefill")
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang" / "request_manager.py"
    spec = importlib.util.spec_from_file_location("request_metrics_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class LegacyCollector:
    """An older hook signature without speculative or aborted-request hooks."""

    def __init__(self):
        self.labels = {"model_name": "test", "priority": "", "tenant": ""}
        self.finished = []
        self.ttft = []
        self.itl = []

    def observe_one_finished_request(self, **kwargs):
        self.finished.append(kwargs)

    def observe_time_to_first_token(self, labels, value):
        self.ttft.append((labels, value))

    def observe_inter_token_latency(self, labels, interval, tokens):
        self.itl.append((labels, interval, tokens))


class Collector(LegacyCollector):
    def __init__(self):
        super().__init__()
        self.aborted = []

    def observe_one_finished_request(self, *, spec_verify_ct=0, **kwargs):
        self.finished.append({**kwargs, "spec_verify_ct": spec_verify_ct})

    def observe_one_aborted_request(self, labels):
        self.aborted.append(labels)


def _manager_and_state(module, collector=None):
    manager = module.GrpcRequestManager.__new__(module.GrpcRequestManager)
    manager.enable_trace = False  # target tracing field, fixture-only adaptation
    manager.metrics_collector = collector if collector is not None else Collector()
    manager.disaggregation_mode = "null"
    manager._ttft_metric_parameters = inspect.signature(
        manager.metrics_collector.observe_time_to_first_token
    ).parameters
    manager._finished_metric_parameters = inspect.signature(
        manager.metrics_collector.observe_one_finished_request
    ).parameters
    manager._send_to_scheduler = AsyncMock()
    state = module.GrpcReqState(
        request_id="test",
        grpc_context=None,
        out_queue=asyncio.Queue(),
        finished=False,
        event=asyncio.Event(),
        obj=SimpleNamespace(
            stream=True,
            priority=3,
            custom_labels={"tenant": "test-tenant", "not_allowed": "ignored"},
            sampling_params=SimpleNamespace(),
        ),
        time_stats=SimpleNamespace(
            first_token_time=1.0,
            get_first_token_latency=lambda: 0.1,
            get_interval=lambda: 0.2,
            get_e2e_latency=lambda: 0.3,
            set_last_time=lambda: None,
            set_finished_time=lambda ts=None: None,
        ),
    )
    manager.rid_to_state = {state.request_id: state}
    return manager, state


def _output(reason=None, **extra):
    return {
        "finished": True,
        "meta_info": {
            "finish_reason": reason or {"type": "length"},
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "cached_tokens": 2,
        },
        **extra,
    }


@pytest.mark.parametrize(
    "fields,index,expected",
    [
        ({}, 0, 0),
        ({"spec_verify_ct": None}, 0, 0),
        ({"spec_verify_ct": []}, 0, 0),
        ({"spec_verify_ct": [7]}, 0, 7),
        ({"spec_verify_ct": [7]}, 1, 0),
        ({"spec_verify_ct": [7, 9]}, 1, 9),
    ],
)
def test_spec_verify_count_is_optional_and_indexed(manager_mod, fields, index, expected):
    manager, state = _manager_and_state(manager_mod)
    manager._collect_generation_metrics(state, _output(), SimpleNamespace(**fields), index)
    assert manager.metrics_collector.finished[0]["spec_verify_ct"] == expected


def test_legacy_finished_hook_has_no_spec_argument(manager_mod):
    manager, state = _manager_and_state(manager_mod, LegacyCollector())
    manager._collect_generation_metrics(state, _output(), SimpleNamespace(spec_verify_ct=[7]), 0)
    assert "spec_verify_ct" not in manager.metrics_collector.finished[0]


@pytest.mark.parametrize("priority,expected", [(0, "0"), (3, "3"), (-1, "-1"), (None, "")])
def test_priority_is_a_string_and_labels_are_not_mutated(manager_mod, priority, expected):
    manager, state = _manager_and_state(manager_mod)
    state.obj.priority = priority
    manager._collect_generation_metrics(state, _output(), SimpleNamespace(), 0)
    labels = manager.metrics_collector.finished[0]["labels"]
    assert labels == {"model_name": "test", "priority": expected, "tenant": "test-tenant"}
    assert manager.metrics_collector.ttft[0][0] == labels
    assert manager.metrics_collector.labels["priority"] == ""
    assert manager.metrics_collector.labels["tenant"] == ""


@pytest.mark.parametrize("output", [_output({"type": "abort"}), _output(error="failed")])
def test_batch_abort_is_counted_without_completion_metrics(manager_mod, output):
    manager, state = _manager_and_state(manager_mod)
    manager._collect_generation_metrics(state, output, SimpleNamespace(), 0)
    manager._collect_generation_metrics(state, output, SimpleNamespace(), 0)
    assert manager.metrics_collector.aborted == [
        {"model_name": "test", "priority": "3", "tenant": "test-tenant"}
    ]
    assert manager.metrics_collector.finished == []
    assert manager.metrics_collector.ttft == []
    assert manager.metrics_collector.itl == []


def test_local_abort_and_scheduler_ack_are_counted_once(manager_mod):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        assert await manager.abort_request("test")
        assert len(manager.metrics_collector.aborted) == 1
        assert state.finished
        await manager.abort_request("test")
        await manager._handle_abort_req(SimpleNamespace(rid="test", finished_reason=None))
        manager._collect_generation_metrics(state, _output({"type": "abort"}), SimpleNamespace(), 0)
        assert len(manager.metrics_collector.aborted) == 1
        assert not manager.metrics_collector.finished

    asyncio.run(exercise())


@pytest.mark.parametrize("reason", [None, {"type": "abort", "message": "queue full"}])
def test_scheduler_abort_is_counted_once(manager_mod, reason):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        recv = SimpleNamespace(rid="test", finished_reason=reason)
        await manager._handle_abort_req(recv)
        await manager._handle_abort_req(recv)
        assert len(manager.metrics_collector.aborted) == 1
        assert state.finished and state.event.is_set()
        assert not manager.metrics_collector.finished

    asyncio.run(exercise())


@pytest.mark.parametrize("path", ["batch", "local", "scheduler"])
def test_internal_warmup_requests_do_not_record_aborts(manager_mod, path):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        state.log_metrics = False
        if path == "batch":
            manager._collect_generation_metrics(
                state, _output({"type": "abort"}), SimpleNamespace(), 0
            )
        elif path == "local":
            await manager.abort_request("test")
        else:
            await manager._handle_abort_req(SimpleNamespace(rid="test", finished_reason=None))
        assert not manager.metrics_collector.aborted
        assert not manager.metrics_collector.finished

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "request_id,expected",
    [
        ("grpc-user-request", True),
        ("HEALTH_CHECK_123", False),
        ("WARMUP_123", False),
        ("smg-warmup-0", False),
    ],
)
def test_internal_request_ids_disable_metrics(manager_mod, request_id, expected):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        manager.rid_to_state = {request_id: state}
        manager._req_stats_init = lambda obj, context: state
        manager.record_request_for_crash_dump = lambda obj: None
        state.request_id = request_id
        state.obj.rid = request_id
        state.time_stats.finished_time = 1.0
        state.time_stats.set_api_server_dispatch_time = lambda: None
        state.time_stats.set_api_server_dispatch_finish_time = lambda: None
        state.out_queue.put_nowait(_output())
        async for _ in manager._handle_single_request(state.obj, request_id):
            pass
        assert state.log_metrics is expected

    asyncio.run(exercise())


@pytest.mark.parametrize("path", ["local", "scheduler"])
def test_completed_requests_are_not_counted_as_aborted(manager_mod, path):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        state.finished = True
        if path == "local":
            await manager.abort_request("test")
        else:
            await manager._handle_abort_req(SimpleNamespace(rid="test", finished_reason=None))
        assert not manager.metrics_collector.aborted

    asyncio.run(exercise())


@pytest.mark.parametrize("collector", [None, LegacyCollector])
def test_abort_works_with_metrics_disabled_or_legacy_collector(manager_mod, collector):
    async def exercise():
        manager, state = _manager_and_state(manager_mod)
        manager.metrics_collector = collector() if collector else None
        assert await manager.abort_request("test")
        await manager._handle_abort_req(SimpleNamespace(rid="test", finished_reason=None))
        manager._collect_generation_metrics(state, _output({"type": "abort"}), SimpleNamespace(), 0)
        assert state.finished

    asyncio.run(exercise())


def test_normal_streaming_metrics_remain_token_weighted(manager_mod):
    manager, state = _manager_and_state(manager_mod)
    first = _output()
    first["finished"] = False
    first["meta_info"]["completion_tokens"] = 1
    manager._collect_generation_metrics(state, first, SimpleNamespace(), 0)
    manager._collect_generation_metrics(state, _output(), SimpleNamespace(spec_verify_ct=[2]), 0)
    collector = manager.metrics_collector
    assert len(collector.ttft) == 1
    assert collector.itl[0][1:] == (0.2, 3)
    assert len(collector.finished) == 1
    assert collector.finished[0]["prompt_tokens"] == 10
    assert collector.finished[0]["generation_tokens"] == 4
    assert collector.finished[0]["cached_tokens"] == 2
    assert collector.finished[0]["e2e_latency"] == 0.3
    assert not collector.aborted
