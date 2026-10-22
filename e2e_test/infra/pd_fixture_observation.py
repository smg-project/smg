"""Allocate one explicit class-fixture attempt, without claiming pytest retry IDs."""

import os
import uuid

from .constants import ConnectionMode
from .model_specs import get_model_spec
from .pd_worker_observation import WorkerObservation
from .worker import Worker


def fixture_observations(config, *, model_id, engine, mode, pools):
    if not isinstance(config, dict) or set(config) != {"run", "attempt", "trace_dir"}:
        raise ValueError("observation config requires run, attempt and trace_dir")
    if engine != "vllm" or mode != ConnectionMode.HTTP:
        raise ValueError("observed fixture requires effective vLLM HTTP mode")
    WorkerObservation(config["run"], config["attempt"], "validate", config["trace_dir"])
    attempt = f"{config['attempt']}-fixture-{uuid.uuid4().hex}"
    result = {}
    for role, count, tp, backends, args, extra_env in pools:
        if extra_env is not None and (
            not isinstance(extra_env, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in extra_env.items())
        ):
            raise ValueError("observed fixture environment requires string keys and values")
        if any(key.startswith("VLLM_DP_") for key in {*os.environ, *(extra_env or {})}):
            raise ValueError("observed fixture refuses DP environment overrides")
        if type(count) is not int or count < 1 or type(tp) is not int or tp != 1:
            raise ValueError(
                "observed fixture supports declared TP1 only; runtime rank is unverified"
            )
        merged = [*get_model_spec(model_id).get("vllm_args", []), *(args or [])]
        normalized = [arg.split("=", 1)[0].replace("_", "-") for arg in merged]
        if any(
            arg in ("-tp", "-dp", "-pp", "-dcp")
            or any(
                flag.startswith(arg) or arg.startswith(flag)
                for flag in (
                    "--data-parallel",
                    "--tensor-parallel",
                    "--pipeline-parallel",
                    "--decode-context-parallel",
                    "--nnodes",
                    "--node-rank",
                    "--distributed-executor-backend",
                )
            )
            for arg in normalized
            if arg.startswith("-")
        ):
            raise ValueError("observed fixture refuses parallelism overrides")
        contexts = []
        for index, backend in enumerate(backends):
            context = WorkerObservation(
                config["run"], attempt, f"{role.value}-{index}", config["trace_dir"]
            )
            # Fixed dummy ports make command preflight side-effect free, including MoRI.
            probe = Worker(
                model_id,
                engine,
                0,
                [],
                mode=mode,
                worker_type=role,
                tp=tp,
                kv_backend=backend,
                extra_engine_args=args,
                moriio_handshake_port=1,
                moriio_notify_port=2,
            )
            context.validate_launch(
                engine=engine, mode=mode, worker_type=role, command=probe._build_cmd()
            )
            contexts.append(context)
        if len(contexts) != count:
            raise ValueError("observation pool size mismatch")
        result[role] = contexts
    return result
