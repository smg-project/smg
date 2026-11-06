"""SGLang 0.5.22 request TPOT parity; no engine or GPU imports required."""

import __future__

import ast
import asyncio
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_sglang_request_metrics import Collector, _manager_and_state
from test_sglang_request_metrics import manager_mod as _manager_fixture

manager_mod = _manager_fixture


class ClockStats:
    def __init__(self, clock):
        self.clock = clock
        self.created_time = 10.0
        self.first_token_time = self.last_time = self.finished_time = 0.0

    def set_first_token_time(self, ts=None):
        self.first_token_time = self.clock[0] if ts is None else ts
        self.last_time = self.first_token_time

    def set_last_time(self, ts=None):
        self.last_time = self.clock[0] if ts is None else ts

    def set_finished_time(self, ts=None):
        self.finished_time = self.clock[0] if ts is None else ts

    def get_first_token_latency(self):
        return self.first_token_time - self.created_time

    def get_interval(self):
        return self.clock[0] - self.last_time

    def get_e2e_latency(self):
        return self.finished_time - self.created_time

    def get_time_per_output_token(self, tokens):
        delta = self.finished_time - self.first_token_time
        return delta / (tokens - 1) if tokens > 1 and delta > 0 else None


def batch(tokens, finish=None):
    return SimpleNamespace(
        rids=["test"],
        prompt_tokens=[8],
        completion_tokens=[tokens],
        cached_tokens=[3],
        output_ids=[[] if tokens == 0 else [7]],
        finished_reasons=[finish],
        cached_tokens_details=[{"device": 2, "host": 1}],
        spec_verify_ct=[2],
    )


class ModernCollector(Collector):
    def observe_one_finished_request(self, *, time_per_output_token=None, **kwargs):
        super().observe_one_finished_request(time_per_output_token=time_per_output_token, **kwargs)


def setup(module, collector=None, stream=True):
    rm, state = _manager_and_state(module, collector or ModernCollector())
    rm.enable_trace = False
    clock = [10.0]
    state.time_stats = ClockStats(clock)
    state.obj.stream = stream
    return rm, state, clock


def output(rm, clock, steps):
    async def scenario():
        for ts, tokens, finish in steps:
            clock[0] = ts
            await rm._handle_batch_output(batch(tokens, finish))

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", ["stop", "length"])
def test_request_tpot_uses_decode_duration_per_request(manager_mod, stream, finish):
    rm, state, clock = setup(manager_mod, stream=stream)
    output(
        rm,
        clock,
        [
            (12, 2, None),
            (16, 4, None),
            (18, 4, None),
            (22, 7, {"type": finish}),
            (24, 7, {"type": finish}),
        ],
    )
    assert len(rm.metrics_collector.finished) == 1
    assert rm.metrics_collector.finished[0]["time_per_output_token"] == pytest.approx(10 / 6)
    assert [row[1:] for row in rm.metrics_collector.itl] == [(4, 2), (6, 3)]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 8])
def test_first_final_has_no_artificial_request_tpot(manager_mod, stream, tokens):
    rm, state, clock = setup(manager_mod, stream=stream)
    ticks = iter([12.0, 12.01])
    state.time_stats.set_first_token_time = lambda: ClockStats.set_first_token_time(
        state.time_stats, ts=next(ticks)
    )
    state.time_stats.set_finished_time = lambda ts=None: ClockStats.set_finished_time(
        state.time_stats, ts=next(ticks) if ts is None else ts
    )
    output(rm, clock, [(12, tokens, {"type": "stop"})])
    if tokens == 0:
        assert state.time_stats.first_token_time == 0.0
        assert not rm.metrics_collector.ttft
    else:
        assert state.time_stats.first_token_time == state.time_stats.finished_time
    assert rm.metrics_collector.finished[0]["time_per_output_token"] is None
    assert not rm.metrics_collector.itl


def test_zero_token_output_does_not_start_generation_timing(manager_mod):
    rm, state, clock = setup(manager_mod)
    output(rm, clock, [(12, 0, None)])
    assert state.time_stats.first_token_time == 0.0
    assert not state.ttft_observed
    assert not rm.metrics_collector.ttft

    output(rm, clock, [(16, 2, None), (20, 3, {"type": "stop"})])
    assert state.time_stats.first_token_time == 16.0
    assert rm.metrics_collector.ttft[0][1] == 6.0
    assert [row[1:] for row in rm.metrics_collector.itl] == [(4.0, 1)]
    assert rm.metrics_collector.finished[0]["time_per_output_token"] == 2.0


def test_prefill_has_no_request_tpot(manager_mod):
    rm, state, clock = setup(manager_mod)
    rm.disaggregation_mode = "prefill"
    output(rm, clock, [(12, 1, None), (16, 3, {"type": "stop"})])
    assert rm.metrics_collector.finished[0]["time_per_output_token"] is None
    assert not rm.metrics_collector.ttft


def test_abort_has_no_success_or_request_tpot(manager_mod):
    rm, state, clock = setup(manager_mod)
    output(rm, clock, [(12, 1, None), (16, 3, {"type": "abort"})])
    assert not rm.metrics_collector.finished
    assert len(rm.metrics_collector.aborted) == 1


@pytest.mark.parametrize("reason", ["error", "unknown"])
def test_non_success_finish_has_no_request_tpot(manager_mod, reason):
    rm, state, clock = setup(manager_mod)
    output(rm, clock, [(12, 1, None), (16, 3, {"type": reason})])
    assert rm.metrics_collector.finished[0]["time_per_output_token"] is None


def test_legacy_collector_needs_no_tpot_method(manager_mod):
    rm, state, clock = setup(manager_mod, Collector())
    state.time_stats.get_time_per_output_token = None
    output(rm, clock, [(12, 1, None), (16, 3, {"type": "stop"})])
    assert "time_per_output_token" not in rm.metrics_collector.finished[0]


def test_real_0522_collector_records_request_tpot(manager_mod):
    source = os.environ.get("SGLANG_METRICS_SOURCE")
    if not source:
        pytest.skip("set SGLANG_METRICS_SOURCE to the exact SGLang observability source")
    prom = pytest.importorskip("prometheus_client")
    root = Path(source)
    tree = ast.parse((root / "metrics_collector.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in ("_StatLoggerDIMixin", "TokenizerMetricsCollector")
    ]
    assert len(tree.body) == 2
    obs = SimpleNamespace(prompt_tokens_buckets=None, generation_tokens_buckets=None)
    ns = {
        "get_observability": lambda: obs,
        "generate_buckets": runpy.run_path(str(root / "utils.py"))["generate_buckets"],
    }
    exec(
        compile(
            tree,
            str(root / "metrics_collector.py"),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    before = set(prom.REGISTRY._collector_to_names)
    try:
        labels = {"model_name": "model", "engine_type": "unified", "priority": "", "tenant": ""}
        c = ns["TokenizerMetricsCollector"](
            server_args=SimpleNamespace(),
            labels=labels,
            bucket_time_to_first_token=[1, 2],
            bucket_e2e_request_latency=[2, 4],
            bucket_inter_token_latency=[0.1, 1],
        )
        rm, state, clock = setup(manager_mod, c)
        output(
            rm,
            clock,
            [
                (12, 2, None),
                (16, 4, None),
                (18, 4, None),
                (22, 7, {"type": "stop"}),
                (24, 7, {"type": "stop"}),
            ],
        )
        base = {**labels, "priority": "3", "tenant": "test-tenant"}
        stream = {**base, "is_streaming": "true"}
        c.histogram_request_time_per_output_token.labels(**stream)

        def value(name, labels):
            return prom.REGISTRY.get_sample_value("sglang:" + name, labels)

        assert value("time_to_first_token_seconds_count", stream) == 1
        assert value("inter_token_latency_seconds_count", base) == 5
        assert value("inter_token_latency_seconds_sum", base) == 10
        assert value("request_time_per_output_token_seconds_count", stream) == 1
        assert value("request_time_per_output_token_seconds_sum", stream) == pytest.approx(10 / 6)
        assert value("num_requests_total", stream) == 1
        assert value("generation_tokens_total", stream) == 7
        assert value("spec_verify_calls_total", base) == 2
        assert value("cached_tokens_total", {**base, "cache_source": "device"}) == 2
        assert value("cached_tokens_total", {**base, "cache_source": "host"}) == 1
    finally:
        for collector in set(prom.REGISTRY._collector_to_names) - before:
            prom.REGISTRY.unregister(collector)
