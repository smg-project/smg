"""Benchmark-specific fixtures."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from infra import GPUMonitor, should_monitor_gpu, terminate_process

from .results import BenchmarkResult

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "ghcr.io/moirai-internal/genai-bench:0.0.4"


def _build_command(
    router_url: str,
    model_path: str,
    experiment_folder: str,
    num_concurrency: int | None,
    traffic_scenario: str | None,
    max_requests: int,
    task: str = "text-to-text",
    max_time_per_run: int = 3,
    server_engine: str | None = None,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
) -> list[str]:
    """Build genai-bench command via docker run."""
    image = os.environ.get("GENAI_BENCH_IMAGE", _DEFAULT_IMAGE)
    base_dir = str(Path.cwd())

    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-v",
        f"{base_dir}:{base_dir}",
        "-w",
        base_dir,
    ]

    # Mount local model directory if configured (e.g. /raid/models)
    local_model_path = os.environ.get("ROUTER_LOCAL_MODEL_PATH")
    if local_model_path:
        cmd.extend(["-v", f"{local_model_path}:{local_model_path}"])

    # Mount host HF cache into container so genai-bench reuses tokenizers
    # already downloaded by sglang workers instead of downloading from HF
    # (HF downloads inside ephemeral containers hang intermittently).
    hf_home = os.environ.get("HF_HOME", os.path.join(Path.home(), ".cache", "huggingface"))
    if os.path.isdir(hf_home):
        cmd.extend(["-v", f"{hf_home}:{hf_home}", "-e", f"HF_HOME={hf_home}"])

    # Pass through environment variables the container may need
    for var in ("HF_TOKEN",):
        if os.environ.get(var):
            cmd.extend(["-e", var])

    # Use local tokenizer path if model is available on disk (avoids slow HF download)
    tokenizer_path = model_path
    if local_model_path:
        local_tokenizer = os.path.join(local_model_path, model_path)
        if os.path.isdir(local_tokenizer):
            tokenizer_path = local_tokenizer

    cmd.extend(
        [
            image,
            "benchmark",
            "--api-backend",
            "openai",
            "--api-base",
            router_url,
            "--api-key",
            "dummy-token",
            "--api-model-name",
            model_path,
            "--model-tokenizer",
            tokenizer_path,
            "--task",
            task,
            "--max-requests-per-run",
            str(max_requests),
            "--max-time-per-run",
            str(max_time_per_run),
            "--experiment-folder-name",
            experiment_folder,
            "--experiment-base-dir",
            base_dir,
        ]
    )
    if num_concurrency is not None:
        cmd.extend(["--num-concurrency", str(num_concurrency)])
    if traffic_scenario is not None:
        cmd.extend(["--traffic-scenario", traffic_scenario])
    if server_engine:
        cmd.extend(["--server-engine", server_engine])
    if gpu_type:
        cmd.extend(["--server-gpu-type", gpu_type])
    if gpu_count:
        cmd.extend(["--server-gpu-count", str(gpu_count)])
    log_dir = os.environ.get("E2E_LOG_DIR")
    if log_dir:
        cmd.extend(["--log-dir", log_dir])
    return cmd


def _find_results(experiment_folder: str, timeout: int = 10) -> list[Path]:
    """Find benchmark result JSON files."""
    base = Path.cwd()
    folder = base / experiment_folder

    if not folder.is_dir():
        # Search for folder
        for p in base.rglob(experiment_folder):
            if p.is_dir() and p.name == experiment_folder:
                folder = p
                break

    if not folder.is_dir():
        raise AssertionError(f"Experiment folder not found: {experiment_folder}")

    # Wait for JSON results
    for _ in range(timeout):
        files = [
            p
            for p in folder.rglob("*.json")
            if "experiment_metadata" not in p.name and "gpu_utilization" not in p.name
        ]
        if files:
            return files
        time.sleep(1)

    raise AssertionError(f"No JSON results found in {folder}")


def _cleanup_procs(procs: list, drain_delay: int) -> None:
    """Terminate processes gracefully."""
    if not procs:
        return
    if drain_delay > 0:
        time.sleep(drain_delay)
    for p in procs:
        try:
            proc = getattr(p, "proc", p)
            if isinstance(proc, subprocess.Popen):
                terminate_process(proc)
        except Exception:
            pass
    time.sleep(2)


def _thresholds_enforced() -> bool:
    """Threshold misses fail the test only when BENCH_ENFORCE_THRESHOLDS is set."""
    value = os.environ.get("BENCH_ENFORCE_THRESHOLDS", "").strip().lower()
    return value in ("1", "true", "yes")


def _report_threshold_misses(experiment: str, misses: list[str]) -> None:
    """Surface threshold misses without failing the run by default.

    The thresholds are absolute latencies and throughputs measured on a shared
    runner pool and sit inside its noise band, so a miss is advisory: it is
    logged, annotated on the workflow run, and appended to the step summary.
    BENCH_ENFORCE_THRESHOLDS=1 turns misses back into test failures.
    """
    if not misses:
        return
    for miss in misses:
        logger.warning("genai-bench[%s] threshold miss: %s", experiment, miss)
        print(f"::warning title=Benchmark threshold miss ({experiment})::{miss}", flush=True)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(f"### Benchmark threshold misses: {experiment}\n\n")
            f.writelines(f"- {miss}\n" for miss in misses)
            f.write("\n")
    if _thresholds_enforced():
        pytest.fail(f"{experiment}: " + "; ".join(misses))


@pytest.fixture(scope="session")
def genai_bench_runner():
    """Run genai-bench and validate metrics.

    Usage:
        def test_perf(setup_backend, genai_bench_runner):
            backend, model_path, client, gateway = setup_backend
            genai_bench_runner(
                router_url=gateway.base_url,
                model_path=model_path,
                experiment_folder="benchmark_results",
                thresholds={"ttft_mean_max": 5, "gpu_util_p50_min": 99},
            )
    """

    def _run(
        *,
        router_url: str,
        model_path: str,
        experiment_folder: str,
        thresholds: dict | None = None,
        timeout_sec: int | None = None,
        num_concurrency: int | None = 32,
        traffic_scenario: str | None = "D(4000,100)",
        max_requests_per_run: int | None = None,
        task: str = "text-to-text",
        max_time_per_run: int = 3,
        server_engine: str | None = None,
        gpu_type: str | None = None,
        gpu_count: int | None = None,
        kill_procs: list | None = None,
        drain_delay_sec: int = 6,
    ) -> None:
        # Clean previous results
        exp_dir = Path.cwd() / experiment_folder
        if exp_dir.exists():
            shutil.rmtree(exp_dir, ignore_errors=True)
        # Create the folder before genai-bench does: the container runs as root
        # and a root-owned folder is unwritable for the host-side GPU monitor.
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Build and run command
        max_requests = max_requests_per_run or (num_concurrency or 32) * 5
        timeout = timeout_sec or int(os.environ.get("GENAI_BENCH_TEST_TIMEOUT", "240"))

        cmd = _build_command(
            router_url,
            model_path,
            experiment_folder,
            num_concurrency,
            traffic_scenario,
            max_requests,
            task=task,
            max_time_per_run=max_time_per_run,
            server_engine=server_engine,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
        )

        logger.info("Running genai-bench command: %s", " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            pytest.fail("docker not found — is Docker installed?")
        except OSError as e:
            pytest.fail(f"Failed to start genai-bench container: {e}")

        # Start GPU monitor if needed
        gpu_monitor: GPUMonitor | None = None
        if should_monitor_gpu(thresholds):
            interval = float(os.environ.get("GPU_UTIL_SAMPLE_INTERVAL", "2.0"))
            gpu_monitor = GPUMonitor(output_dir=exp_dir, interval=interval)
            gpu_monitor.start(target_pid=proc.pid)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            logger.error("genai-bench timed out after %ds", timeout)

        # Fail immediately if genai-bench failed
        if proc.returncode != 0:
            pytest.fail(
                f"genai-bench failed with exit code {proc.returncode}. "
                f"Check CI logs above for details."
            )

        misses: list[str] = []
        try:
            # Parse results; a run without results is broken and still fails
            for path in _find_results(experiment_folder):
                result = BenchmarkResult.from_json(path)
                result.log(experiment_folder, logger)
                if thresholds:
                    misses.extend(
                        f"{path.name}: {miss}" for miss in result.threshold_misses(thresholds)
                    )

            if gpu_monitor:
                gpu_monitor.stop()
                gpu_monitor.log_summary()
                misses.extend(gpu_monitor.threshold_misses(thresholds))

        finally:
            _cleanup_procs(kill_procs or [], drain_delay_sec)
            if gpu_monitor:
                gpu_monitor.stop(timeout=2)

        _report_threshold_misses(experiment_folder, misses)

    return _run
