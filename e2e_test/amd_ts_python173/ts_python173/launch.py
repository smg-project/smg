"""Explicit Python worker extension selection; no mutation of normal harness paths."""
from .process import require

EXTENSION = "smg_grpc_servicer.vllm.request_profile_worker.RequestProfileWorkerExtension"


def install():
    from infra.worker import Worker
    original = Worker._build_vllm_grpc_cmd

    def build(worker, model_path, tp_size, spec):
        require(worker.engine == "vllm" and tp_size == 1, "Python173 needs vLLM TP1")
        cmd = original(worker, model_path, tp_size, spec)
        for flag, expected in (("--worker-extension-cls", EXTENSION), ("--servicer-impl", "python")):
            require(cmd.count(flag) <= 1, "duplicate " + flag)
            if flag in cmd:
                index = cmd.index(flag)
                require(index + 1 < len(cmd) and cmd[index + 1] == expected, "refusing to replace " + flag)
            else:
                cmd.extend([flag, expected])
        return cmd

    Worker._build_vllm_grpc_cmd = build
    return original


def restore(original):
    from infra.worker import Worker
    Worker._build_vllm_grpc_cmd = original
