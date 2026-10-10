"""Nightly comprehensive benchmark tests.

Runs models using genai-bench default scenarios and concurrency levels
(nightly-benchmark-amd.yml). No performance thresholds — results are uploaded
as artifacts for tracking over time.

Each model has Single (1 worker) and Multi (N workers) classes, both
parametrized with http and grpc backends. The workflow matrix crosses
model × variant (single/multi × sglang/vllm); both protocols run for
all runtimes.

genai-bench defaults (omitted flags):
  - Concurrency: [1, 2, 4, 8, 16, 32, 64, 128, 256]
  - Text scenarios: N(480,240)/(300,150), D(100,100), D(100,1000),
                     D(2000,200), D(7800,200)
  - Embedding scenarios: E(64), E(128), E(256), E(512), E(1024)
"""

import os

import pytest
from infra import get_runtime
from infra.model_specs import get_model_spec

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_MAX_REQUESTS = 300
_MAX_TIME_PER_RUN = 10  # seconds per scenario×concurrency combo
_TIMEOUT_SEC = 1440 * 60  # match workflow timeout-minutes

_TEST_MODE = os.environ.get("BENCH_TEST_MODE", "0") == "1"
_TEST_NUM_CONCURRENCY = 1
_TEST_TRAFFIC_SCENARIO = "D(100,100)"
_TEST_MAX_REQUESTS = 10


def _run_nightly(setup_backend, genai_bench_runner, model_id, worker_count=1, **kwargs):
    """Run nightly benchmark for a model with genai-bench defaults."""
    backend, model_path, client, gateway = setup_backend

    # Get runtime and GPU info for metadata
    runtime = get_runtime()  # sglang or vllm from E2E_RUNTIME env var
    # Map to genai-bench expected case (SGLang, vLLM)
    runtime_display = {"sglang": "SGLang", "vllm": "vLLM"}.get(runtime, runtime)
    gpu_type = os.environ.get("GPU_TYPE", "H200")

    # Get tp (tensor parallelism) from model spec - this is GPUs per worker
    model_spec = get_model_spec(model_id)
    tp_per_worker = model_spec.get("tp", 1)

    # Determine worker type and GPU count
    worker_type = "single" if worker_count == 1 else "multi"
    # Total GPU count = tp * workers
    gpu_count = tp_per_worker * worker_count

    # Keep folder names filesystem-safe while retaining the canonical HF model id in metadata.
    safe_model_id = model_id.replace("/", "__")
    experiment_folder = f"nightly_{safe_model_id}_{backend}_{runtime}_{worker_type}"

    if _TEST_MODE:
        genai_bench_runner(
            router_url=gateway.base_url,
            model_path=model_path,
            experiment_folder=experiment_folder,
            num_concurrency=_TEST_NUM_CONCURRENCY,
            traffic_scenario=_TEST_TRAFFIC_SCENARIO,
            max_requests_per_run=_TEST_MAX_REQUESTS,
            timeout_sec=600,
            server_engine=runtime_display,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            **kwargs,
        )
        return

    genai_bench_runner(
        router_url=gateway.base_url,
        model_path=model_path,
        experiment_folder=experiment_folder,
        num_concurrency=None,  # use genai-bench defaults
        traffic_scenario=None,  # use genai-bench defaults
        max_requests_per_run=_MAX_REQUESTS,
        max_time_per_run=_MAX_TIME_PER_RUN,
        timeout_sec=_TIMEOUT_SEC,
        server_engine=runtime_display,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Model configurations: (model_id, class_name_fragment, multi_workers, backends, extra_kwargs)
# backends: list of backends to test (default: ["http", "grpc"])
# ---------------------------------------------------------------------------

_NIGHTLY_MODELS: list[tuple[str, str, int, list[str], dict]] = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama8b", 4, ["http", "grpc"], {}),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen7b", 4, ["http", "grpc"], {}),
    ("Qwen/Qwen3-30B-A3B", "Qwen30b", 4, ["http", "grpc"], {}),
    ("openai/gpt-oss-20b", "GptOss20b", 1, ["http", "grpc"], {}),
    (
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "Llama4Scout",
        1,
        ["http", "grpc"],
        {},
    ),
]


# ---------------------------------------------------------------------------
# Dynamic test class generation
# ---------------------------------------------------------------------------


def _make_test_class(model_id, worker_count, backends, extra_kwargs):
    """Create a nightly benchmark test class for a model/worker configuration."""
    # Capture worker_count in closure for the test method
    _worker_count = worker_count

    @pytest.mark.nightly
    @pytest.mark.e2e
    @pytest.mark.model(model_id)
    @pytest.mark.workers(count=worker_count)
    @pytest.mark.gateway(policy="round_robin", log_level="debug", log_dir="nightly_gateway_logs")
    @pytest.mark.parametrize("setup_backend", backends, indirect=True)
    class _NightlyTest:
        def test_nightly_perf(self, setup_backend, genai_bench_runner):
            _run_nightly(
                setup_backend,
                genai_bench_runner,
                model_id,
                worker_count=_worker_count,
                **extra_kwargs,
            )

    return _NightlyTest


for _model_id, _name, _multi_workers, _backends, _extra in _NIGHTLY_MODELS:
    for _suffix, _count in [("Single", 1), ("Multi", _multi_workers)]:
        _cls_name = f"TestNightly{_name}{_suffix}"
        _cls = _make_test_class(_model_id, _count, _backends, _extra)
        _cls.__name__ = _cls_name
        _cls.__qualname__ = _cls_name
        globals()[_cls_name] = _cls

# Clean up loop variables from module namespace
del _model_id, _name, _multi_workers, _backends, _extra, _suffix, _count, _cls_name, _cls
