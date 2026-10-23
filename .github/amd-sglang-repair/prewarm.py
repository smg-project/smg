"""One bounded original-worker prewarm; original suite startup limits stay intact."""

import importlib.util
import json
import os
import time
from pathlib import Path

from cold_start_evidence import Collector, guard_environment, require

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HEALTH_SECONDS = 480
IMAGE = "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"


def execute(start, stop, collector, mode, output, clock=time.monotonic):
    began = clock()
    workers = []
    errors = []
    result = {
        "model": MODEL,
        "engine": "sglang",
        "mode": "grpc",
        "health_limit_seconds": HEALTH_SECONDS,
        "total_command_limit_seconds": 600,
        "MAX_JOBS": os.environ.get("MAX_JOBS"),
        "CARGO_BUILD_JOBS": os.environ.get("CARGO_BUILD_JOBS"),
        "ready": False,
        "stop_attempted": False,
        "stop_returned": False,
        "accepted": False,
    }

    def capture(label):
        try:
            collector.snapshot(label, workers[0] if workers else None)
        except Exception as error:
            errors.append({"label": label, "error": repr(error)})

    try:
        collector.sources()
        capture("before")
        workers = start(
            MODEL, "sglang", mode=mode, count=1, wait_ready=False, log_dir="e2e-logs/cold-start"
        )
        require(len(workers) == 1, "unexpected prewarm topology")
        require(workers[0].gpu_ids == [0], "unexpected prewarm GPU slice")
        result["worker"] = {
            "pid": workers[0].process.pid,
            "gpu_ids": workers[0].gpu_ids,
            "command": workers[0]._build_cmd(),
            "port": workers[0].port,
        }
        remaining = int(HEALTH_SECONDS - (clock() - began))
        require(remaining > 0, "prewarm health allowance exhausted before wait")
        workers[0].wait_ready(remaining)
        result["ready_after_seconds"] = clock() - began
        result["ready"] = True
        capture("ready")
        try:
            result["native_cache"] = collector.native_identity()
        except Exception as error:
            errors.append({"label": "native-cache", "error": repr(error)})
    except BaseException as error:
        result["error"] = repr(error)
        capture("failed-before-stop")
        raise
    finally:
        try:
            result["stop_attempted"] = bool(workers)
            stop(workers)
            result["stop_returned"] = True
        finally:
            capture("after-stop")
            result["evidence_errors"] = errors
            result["total_seconds"] = clock() - began
            output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    guard_environment(os.environ)
    require(os.environ.get("SMG_QA_IMAGE") == IMAGE, "unexpected prewarm image")
    budget = json.loads(Path(os.environ["SMG_SGLANG_REPAIR_BUDGET_FILE"]).read_text())
    # 600 is an admission reserve for the unchanged suites, not a completion claim.
    require(budget["diagnostic_deadline_epoch"] - time.time() >= 1200, "insufficient core budget")
    from infra import ConnectionMode, start_workers, stop_workers
    from infra.model_specs import get_model_spec
    from run_tests import cached_models

    cache = cached_models(
        {MODEL: get_model_spec(MODEL)["model"]}, Path(os.environ["HF_HOME"]) / "hub"
    )
    out = Path("diagnostic-results/cold-start")
    collector = Collector(out)
    spec = importlib.util.find_spec("aiter")
    require(spec is not None and spec.origin, "missing active AITER package")
    require(Path(spec.origin).resolve() == collector.root / "aiter/__init__.py", "AITER origin")
    (out / "model.json").write_text(json.dumps(cache, indent=2) + "\n")
    (out / "environment.json").write_text(
        json.dumps(
            {
                key: os.environ.get(key)
                for key in [
                    "SMG_QA_IMAGE",
                    "MAX_JOBS",
                    "CARGO_BUILD_JOBS",
                    "AITER_LOG_MORE",
                    "AITER_JIT_DIR",
                    "AITER_REBUILD",
                    "GPU_ARCHS",
                    "PYTORCH_ROCM_ARCH",
                    "TORCH_EXTENSIONS_DIR",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "E2E_JOB_MARKER",
                    "GITHUB_RUN_ID",
                    "GITHUB_RUN_ATTEMPT",
                    "RUNNER_NAME",
                ]
            },
            indent=2,
        )
        + "\n"
    )
    execute(start_workers, stop_workers, collector, ConnectionMode.GRPC, out / "result.json")


if __name__ == "__main__":
    main()
