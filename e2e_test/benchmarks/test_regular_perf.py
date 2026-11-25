"""Regular router performance benchmark test."""

import os

import pytest
from fixtures.mock_answers import class_workers


@pytest.mark.e2e
@pytest.mark.workers(count=4)
@pytest.mark.gateway(policy="round_robin")
@pytest.mark.parametrize("setup_backend", ["http", "grpc"], indirect=True)
class TestRegularPerf:
    """Performance benchmark for regular (non-PD) router."""

    def test_regular_perf(self, setup_backend, genai_bench_runner):
        """Run genai-bench against regular router and validate metrics."""
        backend, model_path, client, gateway = setup_backend
        workers = class_workers()
        assert all(worker.mode.value == backend for worker in workers), (
            "Stale worker protocol binding"
        )
        worker_gpu_ids = [gpu for worker in workers for gpu in worker.gpu_ids]
        assert len(workers) == 4 and len(worker_gpu_ids) == 4
        assert len(set(worker_gpu_ids)) == 4, "Benchmark workers must name four distinct GPUs"
        genai_bench_runner(
            router_url=gateway.base_url,
            model_path=model_path,
            gpu_monitor_backend="amdsmi" if os.environ.get("SMG_SGLANG_ROCM_IMAGE") else "nvml",
            worker_gpu_ids=worker_gpu_ids,
            experiment_folder=f"benchmark_round_robin_regular_{backend}",
            # Increase max_requests to ensure benchmark runs long enough for
            # accurate GPU utilization sampling (at least 30+ seconds)
            max_requests_per_run=200,
            thresholds={
                "ttft_mean_max": 0.86,
                "e2e_latency_mean_max": 14,
                "input_throughput_mean_min": 800,
                "output_throughput_mean_min": 12,
                "gpu_util_mean_min": 30,
            },
        )
