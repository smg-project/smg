"""CPU compatibility checks against an explicitly selected vLLM source tree.

SMG_VLLM_SOURCE_ROOT=/path/to/vllm pytest .../test_vllm_rust_ops.py
Engine/configuration imports are stubbed; stats, collectors and HTTP routes are
loaded from that tree, so these checks catch drift in the installed contract.
"""

import dataclasses
import enum
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import msgspec
import pytest
from smg_grpc_servicer.vllm.rust_ops import RustOps


@pytest.fixture
def upstream(monkeypatch):
    """Load public vLLM statistics code on CPU while isolating GPU and plugin imports."""
    root = os.environ.get("SMG_VLLM_SOURCE_ROOT")
    if root is None:
        pytest.skip("set SMG_VLLM_SOURCE_ROOT for the native vLLM ops compatibility check")
    root = Path(root) / "vllm"
    from prometheus_client import REGISTRY, values

    monkeypatch.setattr(values, "ValueClass", values.ValueClass)
    collectors_before = set(REGISTRY._collector_to_names)

    def module(name, **attrs):
        """Install a scoped import stub with its parent package relationship."""
        result = types.ModuleType(name)
        result.__path__ = []
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        if "." in name:
            parent, _, child = name.rpartition(".")
            parent_module = sys.modules.get(parent) or module(parent)
            monkeypatch.setattr(parent_module, child, result, raising=False)
        return result

    def load(name, relative):
        """Execute one selected upstream source module inside the isolated import tree."""
        spec = importlib.util.spec_from_file_location(name, root / relative)
        result = module(name)
        spec.loader.exec_module(result)
        return result

    class Empty:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    @dataclasses.dataclass
    class EmptyStats:
        pass

    class Reason(enum.IntEnum):
        STOP = 0
        LENGTH = 1
        ABORT = 2
        ERROR = 3
        REPETITION = 4

        def __str__(self):
            """Match the wire finish reason string used by the frontend."""
            return self.name.lower()

    class EventType(enum.IntEnum):
        QUEUED = 1
        SCHEDULED = 2
        PREEMPTED = 3

    class Event(msgspec.Struct):
        type: int
        timestamp: float

    module("vllm")
    module("vllm.envs", VLLM_COMPUTE_NANS_IN_LOGITS=False, VLLM_LOG_STATS_INTERVAL=10)
    module("vllm.logger", init_logger=logging.getLogger)
    module("vllm.config", SupportsMetricsInfo=object, VllmConfig=object, SpeculativeConfig=object)
    module("vllm.plugins", STAT_LOGGER_PLUGINS_GROUP="test", load_plugins_by_group=lambda *args: {})
    module("vllm.compilation.cuda_graph", CUDAGraphStat=EmptyStats, CUDAGraphLogging=Empty)
    module(
        "vllm.v1.metrics.perf",
        PerfStats=EmptyStats,
        PerfMetricsLogging=Empty,
        PerfMetricsProm=Empty,
    )
    module(
        "vllm.distributed.ec_transfer.ec_connector.metrics",
        ECConnectorLogging=Empty,
        ECConnectorProm=Empty,
    )
    module(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        KVConnectorLogging=Empty,
        KVConnectorProm=Empty,
    )
    # Wire field names come from the actual engine definitions without importing
    # torch/executors: the AST contains only the schema classes needed here.
    import ast

    tree = ast.parse((root / "v1/engine/__init__.py").read_text())
    schemas = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "EngineCoreRequest",
            "EngineCoreOutput",
            "EngineCoreOutputs",
        }:
            fields = tuple(item.target.id for item in node.body if isinstance(item, ast.AnnAssign))
            schemas[node.name] = NS(__struct_fields__=fields)
    module(
        "vllm.v1.engine",
        **schemas,
        FinishReason=Reason,
        EngineCoreEvent=Event,
        EngineCoreEventType=EventType,
    )
    load("vllm.v1.metrics.utils", "v1/metrics/utils.py")
    load("vllm.v1.metrics.buckets", "v1/metrics/buckets.py")
    module("vllm.v1.request", RequestStatus=object)
    load("vllm.v1.metrics.cache_hit_source", "v1/metrics/cache_hit_source.py")
    load("vllm.v1.spec_decode.metrics", "v1/spec_decode/metrics.py")
    # Execute the native handshake method without importing CUDA/client runtime.
    ready_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngineCoreReadyResponse"
    )
    ready_type = msgspec.defstruct(
        "Ready",
        [
            (item.target.id, object, None)
            for item in ready_node.body
            if isinstance(item, ast.AnnAssign)
        ],
    )
    client_tree = ast.parse((root / "v1/engine/core_client.py").read_text())
    client = next(
        node
        for node in client_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MPClient"
    )
    method = next(
        node
        for node in client.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_ready_response"
    )
    namespace = {"msgspec": msgspec, "EngineCoreReadyResponse": ready_type}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "core_client.py", "exec"), namespace)
    module(
        "vllm.v1.engine.core_client",
        MPClient=NS(_apply_ready_response=namespace["_apply_ready_response"]),
    )
    stats = load("vllm.v1.metrics.stats", "v1/metrics/stats.py")
    module("vllm.outputs", CompletionOutput=object)
    module("vllm.sampling_params", RequestOutputKind=NS(FINAL_ONLY=2), SamplingParams=object)
    load("vllm.v1.engine.parallel_sampling", "v1/engine/parallel_sampling.py")
    load("vllm.v1.metrics.prometheus", "v1/metrics/prometheus.py")
    load("vllm.v1.metrics.loggers", "v1/metrics/loggers.py")
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    yield stats
    for collector in set(REGISTRY._collector_to_names) - collectors_before:
        REGISTRY.unregister(collector)


def config():
    """Build the minimal two-rank configuration consumed by native collectors."""
    return NS(
        parallel_config=NS(data_parallel_size=2, data_parallel_rank=0, data_parallel_size_local=2),
        observability_config=NS(
            show_hidden_metrics=False, kv_cache_metrics=True, custom_histogram_buckets={}
        ),
        model_config=NS(served_model_name="test", max_model_len=128, is_diffusion=False),
        speculative_config=None,
        kv_transfer_config=None,
        watermark_config=None,
        ec_transfer_config=None,
        lora_config=None,
        cache_config=NS(metrics_info=lambda: {"block_size": "16"}),
    )


def samples(text):
    """Compare stable vLLM samples, excluding nondeterministic creation timestamps."""
    from prometheus_client.parser import text_string_to_metric_families

    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name.startswith("vllm:") and not sample.name.endswith("_created")
    }


class Replay:
    def __init__(self, cfg):
        """Keep a deterministic event sequence for a native-collector comparison."""
        self.cfg = cfg
        self.events = []

    def event(self, kind, data, at=100.0):
        """Queue a wire event with a deterministic wall-clock timestamp."""
        if kind in ("output", "add"):
            payload = msgspec.msgpack.encode(data)
        elif kind in ("consume", "abort", "forget_parent"):
            payload = data.encode()
        else:
            payload = json.dumps(data).encode()
        self.events.append({"kind": kind, "payload": list(payload), "at": at})

    def run(self):
        """Replay events through the adapter and collect its resulting native samples."""
        import asyncio
        import tempfile
        from unittest.mock import patch

        from prometheus_client import REGISTRY, generate_latest

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, PROMETHEUS_MULTIPROC_DIR=directory),
        ):
            ops = RustOps(self.cfg, 0, log_stats=True)
            ops.observe("ready_done", b"")
            try:
                for event in self.events:
                    with patch("time.time", return_value=event["at"]):
                        ops.observe(event["kind"], bytes(event["payload"]))
                    assert ops.error is None
                return samples(generate_latest(REGISTRY).decode())
            finally:
                asyncio.run(ops.close())


def native_logger(cfg):
    """Build the reference native Prometheus collector for both engine ranks."""
    from vllm.v1.metrics.loggers import PrometheusStatLogger

    logger = PrometheusStatLogger(cfg, engine_indexes=[0, 1])
    logger.log_engine_initialized()
    return logger


def assert_same(replay):
    """Require identical sample names, labels and values for native and replayed events."""
    from prometheus_client import REGISTRY, generate_latest

    expected = samples(generate_latest(REGISTRY).decode())
    actual = replay.run()
    assert actual.keys() == expected.keys(), (
        actual.keys() - expected.keys(),
        expected.keys() - actual.keys(),
    )
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value), key


@pytest.mark.parametrize("rank", [0, 1])
def test_scheduler_all_standard_and_extension_samples(upstream, rank):
    """Preserve scheduler, cache and speculative-decoding samples for each DP rank."""
    cfg = config()
    cfg.speculative_config = NS(num_speculative_tokens=3)
    replay = Replay(cfg)
    logger = native_logger(cfg)
    from vllm.v1.spec_decode.metrics import SpecDecodingStats

    for i in range(1, 4):
        scheduler = upstream.SchedulerStats(
            num_running_reqs=i,
            num_waiting_reqs=i + 2,
            num_skipped_waiting_reqs=i + 1,
            kv_cache_usage=0.3,
            prefix_cache_stats=upstream.PrefixCacheStats(queries=31, hits=19),
            connector_prefix_cache_stats=upstream.PrefixCacheStats(queries=17, hits=5),
            num_kv_fetch_reqs_by_stage={"waiting_to_start": 2, "in_progress": 1},
            kv_cache_eviction_events=[upstream.KVCacheEvictionEvent(5, 1, (0.2, 0.8))],
            spec_decoding_stats=SpecDecodingStats(
                num_spec_tokens=3,
                num_drafts=5,
                num_draft_tokens=15,
                num_accepted_tokens=8,
                num_accepted_tokens_per_pos=[5, 2, 1],
            ),
        )
        logger.record(scheduler, None, engine_idx=rank)
        replay.event("output", [rank, [], dataclasses.asdict(scheduler), 10 + i])
    assert_same(replay)


@pytest.mark.parametrize("tokens", [[1], [1, 1], [3, 2, 1]])
@pytest.mark.parametrize(
    "finish", ["stop", "length", "error", "abort", "repetition", "repetition_stop"]
)
def test_request_formulas_against_native_vllm(upstream, monkeypatch, tokens, finish):
    """Match native token and timing histograms across batches and terminal reasons."""
    native_finish = "stop" if finish == "repetition_stop" else finish
    replay = Replay(config())
    logger = native_logger(config())
    state = upstream.RequestStateStats(arrival_time=100.0)
    lora = upstream.LoRARequestStates()
    replay.event("add", ["r", [1, 2, 3, 4, 5], None, {"max_tokens": 10}, None, 100.0])
    for i, count in enumerate(tokens):
        at = 101.0 + i * 2
        monkeypatch.setattr(upstream.time, "time", lambda: at)
        iteration = upstream.IterationStats()
        prefill = (
            upstream.PrefillStats(
                num_prompt_tokens=5,
                num_computed_tokens=3,
                num_cached_tokens=2,
                num_local_cached_tokens=1,
                num_external_cached_tokens=1,
            )
            if i == 0
            else None
        )
        events = (
            [NS(type=1, timestamp=9.0), NS(type=2, timestamp=10.0)]
            if i == 0
            else [NS(type=3, timestamp=12.0)]
        )
        output = NS(
            request_id="r",
            new_token_ids=[7] * count,
            prefill_stats=prefill,
            events=events,
            num_nans_in_logits=0,
        )
        iteration.update_from_output(output, 11.0 + i * 2, i == 0, state, lora, None)
        if i == len(tokens) - 1:
            from vllm.v1.engine import FinishReason

            iteration.update_from_finished_request(
                FinishReason[native_finish.upper()], "r", 5, 10, state, 2
            )
            iteration.n_params_iter.append(1)
            iteration.max_num_generation_tokens_iter.append(sum(tokens))
        logger.record(None, iteration, engine_idx=1)
        raw = {
            **vars(output),
            "prefill_stats": dataclasses.asdict(prefill) if prefill else None,
            "events": [vars(e) for e in events],
        }
        if i == len(tokens) - 1:
            raw["finish_reason"] = FinishReason[
                ("repetition" if finish == "repetition_stop" else finish).upper()
            ].value
        replay.event("output", [1, [raw], None, 11.0 + i * 2], at)
        replay.event("consume", "r", at)
        if i < len(tokens) - 1:
            replay.event("processed", ["r", 0], at)
    if finish == "error":
        # Rust surfaces engine ERROR as gRPC Status, without GenerateComplete.
        replay.event("forget_parent", "r", at)
    else:
        replay.event(
            "finish",
            [
                "r",
                0,
                "stop" if finish.startswith("repetition") else finish,
                finish == "repetition_stop",
            ],
            at,
        )
    assert_same(replay)


def test_stop_discards_queued_outputs_and_abort_does_not_finish(upstream, monkeypatch):
    """Count only consumed tokens after a frontend stop and discard aborted requests."""
    replay = Replay(config())
    logger = native_logger(config())
    replay.event("add", ["r", [1], None, {"max_tokens": 8}, None, 100.0])
    replay.event("output", [1, [{"request_id": "r", "new_token_ids": [7]}], None, 12.0])
    replay.event(
        "output",
        [1, [{"request_id": "r", "new_token_ids": [8, 9], "finish_reason": 1}], None, 14.0],
    )
    replay.event("consume", "r", 103.0)
    replay.event("finish", ["r", 0, "stop"], 103.0)
    monkeypatch.setattr(upstream.time, "time", lambda: 103.0)
    state = upstream.RequestStateStats(arrival_time=100.0)
    iteration = upstream.IterationStats()
    iteration.update_from_output(
        NS(
            request_id="r", new_token_ids=[7], prefill_stats=None, events=None, num_nans_in_logits=0
        ),
        12,
        True,
        state,
        upstream.LoRARequestStates(),
        None,
    )
    from vllm.v1.engine import FinishReason

    iteration.update_from_finished_request(FinishReason.STOP, "r", 1, 8, state)
    iteration.n_params_iter.append(1)
    iteration.max_num_generation_tokens_iter.append(1)
    logger.record(None, iteration, engine_idx=1)
    # Native OutputProcessor sees the later batch, ignores the finished request,
    # and records a zero-token engine iteration.
    logger.record(None, upstream.IterationStats(), engine_idx=1)
    replay.event("add", ["aborted", [1], None, {}, None, 100.0])
    replay.event("abort", "aborted")
    assert_same(replay)


def test_lora_and_n_group_completion(upstream, monkeypatch):
    """Keep LoRA bookkeeping and parent sampling counts aligned with native collectors."""
    cfg = config()
    cfg.lora_config = NS(max_loras=4)
    replay = Replay(cfg)
    logger = native_logger(cfg)
    lora = upstream.LoRARequestStates(log_stats=True)
    replay.event("parent", ["p", 2])
    for index in range(2):
        request_id = f"p-{index}"
        request = dict(
            request_id=request_id,
            prompt_token_ids=[1],
            sampling_params={"max_tokens": 8},
            arrival_time=100.0,
            lora_request=["adapter", 1],
        )
        replay.event("add", request)
        state = upstream.RequestStateStats(arrival_time=100.0)
        monkeypatch.setattr(upstream.time, "time", lambda: 103.0)
        iteration = upstream.IterationStats()
        output = NS(
            request_id=request_id,
            new_token_ids=[7],
            prefill_stats=None,
            events=[NS(type=1, timestamp=10), NS(type=2, timestamp=11)],
            num_nans_in_logits=0,
        )
        iteration.update_from_output(output, 12, True, state, lora, "adapter")
        from vllm.v1.engine import FinishReason

        iteration.update_from_finished_request(FinishReason.STOP, request_id, 1, 8, state)
        lora.request_finished(request_id, "adapter")
        if index == 1:
            iteration.n_params_iter.append(2)
            iteration.max_num_generation_tokens_iter.append(1)
        scheduler = upstream.SchedulerStats()
        lora.update_scheduler_stats(scheduler)
        logger.record(scheduler, iteration, engine_idx=1)
        replay.event(
            "output",
            [
                1,
                [{**vars(output), "events": [vars(e) for e in output.events]}],
                dataclasses.asdict(upstream.SchedulerStats()),
                12,
            ],
            103,
        )
        replay.event("consume", request_id, 103)
        replay.event("finish", ["p", index, "stop"], 103)
    assert_same(replay)


def test_native_http_scrape_and_cleanup(upstream, monkeypatch, tmp_path):
    """Expose multiprocess metrics over HTTP and reject scrapes after collector failure."""
    import asyncio
    import socket

    import aiohttp
    from prometheus_client import Counter

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    observed = []

    async def exercise():
        """Exercise the listener lifecycle with real loopback sockets."""
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        ops = RustOps(config(), port, log_stats=True)
        ops.observe("ready_done", b"")
        Counter("vllm:worker_test", "worker mmap counter").inc(3)
        await ops.start(NS(set_metrics_observer=observed.append))
        try:
            assert observed == [ops.observe]
            async with aiohttp.ClientSession() as session:
                base = f"http://127.0.0.1:{port}"
                async with session.get(base + "/metrics") as response:
                    assert response.status == 200
                    text = await response.text()
                    assert "vllm:worker_test_total 3.0" in text
                    assert "vllm:time_to_first_token_seconds" in text
                for path in ("/start_profile", "/stop_profile"):
                    async with session.post(base + path) as response:
                        assert response.status == 404
                ops.observe("output", b"invalid msgpack")
                assert ops.error is not None
                async with session.get(base + "/metrics") as response:
                    assert response.status == 503
        finally:
            await ops.close()
            await ops.close()
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))

    asyncio.run(exercise())


def test_disabled_stats_and_failed_bind_cleanup(upstream, monkeypatch, tmp_path):
    """Keep disabled statistics inert and release resources after bind failure."""
    import asyncio
    import socket

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))

    async def exercise():
        """Exercise the listener lifecycle with real loopback sockets."""
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            port = sock.getsockname()[1]
            ops = RustOps(config(), port, log_stats=False)
            ops.observe("output", b"invalid msgpack")
            assert ops.error is None and ops.stats is None
            with pytest.raises(OSError):
                await ops.start(NS(set_metrics_observer=lambda _: pytest.fail("stats disabled")))
            await ops.close()
        assert not ops.requests

    asyncio.run(exercise())


def test_scheduler_snapshots_preserve_batch_order(upstream, monkeypatch, tmp_path):
    """A delayed request cannot overwrite a newer scheduler snapshot."""
    import asyncio

    from prometheus_client import REGISTRY

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    ops = RustOps(config(), 0, log_stats=True)
    ops.observe("ready_done", b"")
    try:
        ops.observe("add", msgspec.msgpack.encode(["r", [1], None, {}, None, 100.0]))
        ops.observe(
            "output",
            msgspec.msgpack.encode(
                [
                    1,
                    [{"request_id": "r", "new_token_ids": [7]}],
                    dataclasses.asdict(upstream.SchedulerStats(num_running_reqs=1)),
                    12,
                ]
            ),
        )
        ops.observe(
            "output",
            msgspec.msgpack.encode(
                [
                    1,
                    [],
                    dataclasses.asdict(upstream.SchedulerStats(num_running_reqs=0)),
                    13,
                ]
            ),
        )
        ops.observe("consume", b"r")
        ops.observe("processed", json.dumps(["r", 0]).encode())
        assert ops.error is None
        assert (
            REGISTRY.get_sample_value(
                "vllm:num_requests_running", {"model_name": "test", "engine": "1"}
            )
            == 0
        )
        assert not ops.batches[1]
        ops.observe("abort", b"r")
        assert not ops.requests
    finally:
        asyncio.run(ops.close())


def test_metrics_initialize_after_native_ready_config(upstream, monkeypatch, tmp_path):
    """Apply all rank handshakes before publishing initialized cache descriptors."""
    import asyncio

    from prometheus_client import REGISTRY

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    cfg = config()
    cfg.cache_config.num_gpu_blocks = None
    cfg.cache_config.metrics_info = lambda: {
        key: str(getattr(cfg.cache_config, key))
        for key in (
            "num_gpu_blocks",
            "block_size",
            "mamba_block_size",
            "effective_attention_block_size",
            "kv_cache_size_tokens",
        )
    }
    ops = RustOps(cfg, 0, log_stats=True)
    try:
        assert ops.stats is None
        for rank in range(2):
            ops.observe(
                "ready",
                msgspec.msgpack.encode(
                    {
                        "max_model_len": 64 + rank,
                        "num_gpu_blocks": 10 + rank,
                        "block_size": 32,
                        "mamba_block_size": 64,
                        "effective_attention_block_size": 16 + rank,
                        "kv_cache_size_tokens": 320 + rank,
                        "kv_cache_max_concurrency": 5.0,
                        "data_parallel_rank": rank,
                    }
                ),
            )
            assert ops.error is None
            assert ops.stats is None
        ops.observe("ready_done", b"")
        assert ops.error is None
        assert cfg.model_config.max_model_len == 64
        assert (
            REGISTRY.get_sample_value(
                "vllm:cache_config_info",
                {
                    "engine": "0",
                    "num_gpu_blocks": "21",
                    "block_size": "32",
                    "mamba_block_size": "64",
                    "effective_attention_block_size": "None",
                    "kv_cache_size_tokens": "320",
                },
            )
            == 1
        )
        # Duplicate notification must not register the same collectors twice.
        ops.observe("ready_done", b"")
        assert ops.error is None
    finally:
        asyncio.run(ops.close())


@pytest.mark.parametrize("count", [1, 2])
def test_arrival_includes_frontend_preprocessing(upstream, monkeypatch, tmp_path, count):
    """Single and multi-choice timing starts before frontend media processing."""
    import asyncio

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    ops = RustOps(config(), 0, log_stats=True)
    try:
        ops.observe("ready_done", b"")
        ops.observe("parent", json.dumps(["r", count, 95.0]).encode())
        children = [f"r-{i}" for i in range(count)] if count > 1 else ["r"]
        for child in children:
            ops.observe(
                "add",
                msgspec.msgpack.encode(
                    {
                        "request_id": child,
                        "prompt_token_ids": [1],
                        "sampling_params": {},
                        "arrival_time": 100.0,
                        "external_req_id": "external",
                    }
                ),
            )
            assert ops.requests[child].stats.arrival_time == 95.0
            assert ops.requests[child].external_id == "external"
        ops.observe("forget_parent", b"r")
        assert ops.error is None
        assert not ops.requests and not ops.arrivals and not ops.parents
    finally:
        asyncio.run(ops.close())


def test_control_frames_do_not_require_a_statistics_timestamp(upstream, monkeypatch, tmp_path):
    """Utility and DP control messages share the output socket but carry no metrics batch."""
    import asyncio

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    ops = RustOps(config(), 0, log_stats=True)
    try:
        ops.observe("ready_done", b"")
        for frame in ({"utility_output": {"call_id": 7}}, {"start_wave": 1}):
            ops.observe("output", msgspec.msgpack.encode(frame))
        assert ops.error is None
        assert not ops.batches
    finally:
        asyncio.run(ops.close())
