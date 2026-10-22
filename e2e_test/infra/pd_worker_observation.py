"""Explicit launch context for the test-only vLLM HTTP PD observer."""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from .constants import ConnectionMode, WorkerType


@dataclass(frozen=True)
class WorkerObservation:
    run: str
    attempt: str
    worker: str
    trace_dir: Path

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value
            for value in (self.run, self.attempt, self.worker)
        ):
            raise ValueError("explicit run, attempt and worker are required")
        path = Path(self.trace_dir)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("observer trace directory must exist and be absolute")

    def child_environment(self, base, *, engine, mode, worker_type, command):
        if engine != "vllm" or mode != ConnectionMode.HTTP:
            raise ValueError("observer launch supports vLLM HTTP only")
        roles = {WorkerType.PREFILL: "kv_producer", WorkerType.DECODE: "kv_consumer"}
        if worker_type not in roles:
            raise ValueError("observer launch requires a PD worker")
        configs = []
        for index, argument in enumerate(command):
            flag = argument.split("=", 1)[0]
            normalized = flag.replace("_", "-")
            canonical = "--kv-transfer-config"
            if flag == "-c" or (
                flag.startswith("--")
                and (
                    "--config".startswith(normalized)
                    or normalized.startswith("--config")
                    or (
                        flag != canonical
                        and (canonical.startswith(normalized) or normalized.startswith(canonical))
                    )
                )
            ):
                raise ValueError("unknown worker KV role configuration override")
            if argument == "--kv-transfer-config":
                configs.append(command[index + 1] if index + 1 < len(command) else "")
            elif argument.startswith("--kv-transfer-config="):
                configs.append(argument.split("=", 1)[1])
        if len(configs) != 1:
            raise ValueError("unknown worker KV role configuration")
        config = json.loads(configs[0])
        if not isinstance(config, dict) or config.get("kv_role") != roles[worker_type]:
            raise ValueError("worker KV role does not match observation role")
        from smg_pd_observer import child_environment

        env = dict(base)
        env.update(
            SMG_PD_TRACE_RUN=self.run,
            SMG_PD_TRACE_ATTEMPT=self.attempt,
            SMG_PD_TRACE_WORKER=self.worker,
            SMG_PD_TRACE_ROLE=worker_type.value,
            SMG_PD_TRACE_DIR=str(self.trace_dir),
            SMG_PD_TRACE_WORKER_INSTANCE=uuid.uuid4().hex,
        )
        return child_environment(env, enabled=True)
