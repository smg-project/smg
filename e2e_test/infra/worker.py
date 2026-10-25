"""Simple worker lifecycle management. No pooling, no eviction, no sharing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import IO, Any

import yaml

from .constants import (
    DEFAULT_HOST,
    DEFAULT_STARTUP_TIMEOUT,
    ENV_SHOW_WORKER_LOGS,
    ENV_VLLM_SERVICER_IMPL,
    HEALTH_CHECK_INTERVAL,
    LAUNCH_STAGGER_DELAY,
    MM_PROCESSING_WORKER,
    ConnectionMode,
    WorkerType,
    effective_startup_timeout,
    get_gpu_offset,
    get_mm_processing,
    get_runtime,
    get_sglang_servicer_impl,
    get_tokenspeed_servicer_impl,
    get_vllm_mm_processor,
    get_vllm_servicer_impl,
    get_zmq_engine_count,
    host_port,
    moriio_host,
    sglang_transfer_backend,
    vllm_kv_backend,
    vllm_moriio_backend,
    vllm_moriio_mode,
)
from .model_specs import get_model_spec
from .process_utils import (
    ProcessExitedError,
    detect_ib_device,
    detect_rdma_fabric_devices,
    get_engine_port,
    get_open_port,
    get_open_port_block,
    gpu_compute_apps,
    gpu_memory_used_mib,
    live_child_processes,
    reserve_port,
    wait_for_gpu_memory_release,
    wait_for_health,
    wait_for_process_group_exit,
)

logger = logging.getLogger(__name__)

# How much of a log line the failure text of a dead worker quotes.
LOG_EXCERPT_CHARS = 300
# An exception a process printed (`ValueError: ...`, `torch.OutOfMemoryError: ...`).
_EXCEPTION_LINE = re.compile(r"(?:Error|Exception): ")
# A log line at error level, or a Rust panic, when nothing printed an exception.
_ERROR_LEVEL_LINE = re.compile(r"\b(?:ERROR|CRITICAL|FATAL)\b|panicked at ")
# A line logged below error level: a warning that names an exception (a missing
# optional import, say) is not the cause of the death. The level is read before
# the exception marker only: a level word inside the message (torch's NCCL error
# ends with "run with NCCL_DEBUG=INFO for details") does not lower the line.
_BELOW_ERROR_LEVEL_LINE = re.compile(r"\b(?:DEBUG|INFO|WARNING|WARN)\b")

# The engines whose gRPC server carries no ``grpc.health.v1`` service:
# TRT-LLM, whose upstream server registers its own service and no standard
# health (it answers its own proto's HealthCheck, the one the gateway
# probes). For these alone a ready channel that answers UNIMPLEMENTED counts
# as the worker, until the harness asks their own HealthCheck instead. The
# scripted mock worker hosts the standard service like the engine servicers.
ENGINES_WITHOUT_GRPC_HEALTH = frozenset({"trtllm"})


def _excerpt(line: str) -> str:
    if len(line) <= LOG_EXCERPT_CHARS:
        return line
    return line[: LOG_EXCERPT_CHARS - 3] + "..."


def rotate_previous_log(log_path: str) -> str | None:
    """Move a previous engine's log aside before a new one writes to its path.

    Worker logs are named by port, and a worker is relaunched on its port by
    the restart tests. Opening the same file again with mode "w" erased the
    previous engine's output, which is the one trace a crash investigation
    needs: the nightly PD sweep lost a decode engine's death this way, twice.
    A non-empty file moves to ``<log>.1`` (then ``.2``, ...): the suffix keeps
    it out of the ``worker-*.log`` globs the transfer assertions read and
    inside the directory CI uploads. Returns the rotated path, or ``None``
    when there was nothing to keep.
    """
    try:
        if os.path.getsize(log_path) == 0:
            return None
    except OSError:
        return None
    n = 1
    while os.path.exists(f"{log_path}.{n}"):
        n += 1
    rotated = f"{log_path}.{n}"
    os.replace(log_path, rotated)
    return rotated


@dataclass
class Worker:
    """A single inference worker process."""

    model_id: str
    engine: str  # "sglang", "vllm", "trtllm", or "tokenspeed"
    port: int
    gpu_ids: list[int]
    mode: ConnectionMode = ConnectionMode.HTTP
    worker_type: WorkerType = WorkerType.REGULAR
    bootstrap_port: int | None = None
    nixl_port: int | None = None
    # MoRI-IO KV ports: the handshake port and the first of the notify ports
    # (one per tensor-parallel rank, consecutive). SMG holds them as labels,
    # so a restarted worker keeps them.
    moriio_handshake_port: int | None = None
    moriio_notify_port: int | None = None
    ib_device: str | None = None
    dist_init_addr: str | None = None
    dist_init_port: int | None = None
    log_dir: str | None = None
    extra_engine_args: list[str] | None = None
    # Overrides the model spec's tp, so a PD pair can run asymmetric pools.
    tp: int | None = None
    # KV transfer backend for this PD worker ("nixl", "mooncake" or "moriio");
    # None takes the lane's default, so one fleet can mix transports.
    kv_backend: str | None = None
    # In-engine RL control app port (TokenSpeed gRPC workers). None allocates
    # one at launch; SMG discovers it from the engine's server info.
    rl_control_port: int | None = None
    # Environment for the engine process on top of the infra's own settings
    # (a deployment-injected SMG_PAIRING_PROTOCOL, for instance).
    extra_env: dict[str, str] | None = None
    # Mock engine only: the port of the mock worker's admin API, where a test
    # states the answers the worker's next requests get.
    admin_port: int | None = None
    # Whether the harness picked ``admin_port`` (and releases it at stop).
    _admin_port_auto: bool = field(default=False, repr=False)
    process: subprocess.Popen | None = field(default=None, repr=False)
    # Where the engine's output lands when it is not shown on the terminal.
    log_path: str | None = field(default=None, repr=False)
    _log_file: IO[Any] | None = field(default=None, repr=False)
    # Used memory per GPU just before launch; ``stop`` waits for it to come back.
    _gpu_mem_baseline: dict[int, int] | None = field(default=None, repr=False)
    # The launcher's process group, from its spawn to the end of ``stop``: what
    # is reaped and waited for when the launcher itself is already gone.
    _pgid: int | None = field(default=None, repr=False)
    # Set by ``stop``, cleared by ``start``: the teardown (the reaping, the log
    # handle, the ports) runs once per start, and once for a worker that never
    # started; a second ``stop`` finds nothing to do.
    _torn_down: bool = field(default=False, repr=False)
    _moriio_ports: list[int] = field(default_factory=list, repr=False)

    @property
    def base_url(self) -> str:
        """Base URL for this worker."""
        if self.mode == ConnectionMode.ZMQ:
            # ipc:// worker URL the router binds; the engine dials the tcp
            # handshake port SMG derives from it. Reuse serve's helper so the
            # path format stays in lockstep with the launcher and the router.
            from smg.serve import _zmq_ipc_url

            return _zmq_ipc_url(self.port)
        if self.mode == ConnectionMode.GRPC:
            return f"grpc://{host_port(DEFAULT_HOST, self.port)}"
        return f"http://{host_port(DEFAULT_HOST, self.port)}"

    @property
    def worker_url(self) -> str:
        """Alias for base_url for Gateway compatibility."""
        return self.base_url

    @property
    def http_url(self) -> str:
        """HTTP URL (used for health checks even on gRPC workers)."""
        return f"http://{host_port(DEFAULT_HOST, self.port)}"

    @property
    def admin_url(self) -> str | None:
        """The mock worker's admin API, or ``None`` for an engine that has none."""
        if self.admin_port is None:
            return None
        return f"http://{host_port(DEFAULT_HOST, self.admin_port)}"

    def start(
        self,
        timeout: int = DEFAULT_STARTUP_TIMEOUT,
        wait_ready: bool = True,
    ) -> None:
        """Launch worker process and optionally wait for health.

        Args:
            timeout: Seconds to wait for health check (only used when wait_ready=True).
            wait_ready: If True, block until the worker passes health checks.
                If False, spawn the process and return immediately.
        """
        # Every start arms its teardown, before anything is built or spawned for it.
        self._torn_down = False
        cmd = self._build_cmd()
        env = self._build_env()

        logger.info(
            "Starting %s %s worker for %s on GPUs %s port %d",
            self.engine,
            self.mode.value,
            self.model_id,
            self.gpu_ids,
            self.port,
        )
        logger.debug("Command: %s", " ".join(cmd))

        self._gpu_mem_baseline = gpu_memory_used_mib(self.gpu_ids)
        self.process = self._spawn_process(cmd, env)
        # The launcher leads a session of its own (``start_new_session``), so
        # its pid is its group's id. Kept here: ``os.getpgid`` has no answer
        # for a launcher that has exited and been reaped.
        self._pgid = self.process.pid

        if not wait_ready:
            logger.info(
                "Worker %s spawned at %s (PID %d) — not waiting for health",
                self.model_id,
                self.base_url,
                self.process.pid,
            )
            return

        self.wait_ready(timeout)

    def wait_ready(self, timeout: int = DEFAULT_STARTUP_TIMEOUT) -> None:
        """Block until the spawned worker passes its health check.

        ``start(wait_ready=False)`` followed by this call lets a caller spawn
        several workers and wait for all of them afterwards.
        """
        if self.process is None:
            raise RuntimeError(f"Worker {self.model_id} has not been started")
        if self.mode == ConnectionMode.ZMQ:
            # SMG (the router) binds the ZMQ sockets and this engine dials in;
            # there is no worker port to probe. The gateway's readiness gate
            # (wait_for_workers_ready, started with this worker) covers the
            # engine's load and its exit, so just proceed.
            logger.info(
                "Worker %s spawned at %s (PID %d) — ZMQ readiness gated by the gateway",
                self.model_id,
                self.base_url,
                self.process.pid,
            )
            return
        if self.mode == ConnectionMode.GRPC:
            self._wait_grpc_healthy(timeout)
        else:
            try:
                wait_for_health(
                    self.http_url,
                    timeout=timeout,
                    check_interval=HEALTH_CHECK_INTERVAL,
                    process=self.process,
                )
            except ProcessExitedError:
                # The worker exited before it answered: the same report as the
                # gRPC branch raises, instead of the wait running to its bound.
                raise RuntimeError(self._death_report()) from None

        logger.info(
            "Worker %s healthy at %s (PID %d)",
            self.model_id,
            self.base_url,
            self.process.pid,
        )

    def stop(self) -> None:
        """Terminate the worker and reap everything it started, then release
        what the harness holds for it: the log handle and the reserved ports.

        Also the teardown after the launcher exited on its own (its engine
        died under it, or :meth:`kill_engine` took the engine away): the
        engine's processes may outlive the launcher while they tear down
        their device contexts, and a replacement started on the same GPUs
        inside that window fails its startup memory check against the old
        allocation. So the group and the memory are waited for whether the
        launcher is alive or gone, once per start. A worker that never
        spawned (its launch failed before the spawn) has nothing to wait for;
        its log handle and its ports are released all the same, so a restart
        opens one log file and the port pool keeps its size. A second call
        finds nothing to do.
        """
        if self._torn_down:
            return
        self._torn_down = True
        if self.process is not None and (self._pgid is not None or self.process.poll() is None):
            self._reap(self.process)
        self._close_log_file()
        self._release_ports()

    def _reap(self, process: subprocess.Popen) -> None:
        """Signal the launcher's group while it is alive, then wait for the
        group to empty and for the GPU memory to come back (see :meth:`stop`)."""
        pid = process.pid
        # Workers run in their own session, so the launcher and every engine
        # process share one group, led by the launcher.
        pgid = self._pgid if self._pgid is not None else pid
        self._pgid = None

        if process.poll() is None:
            logger.info("Stopping worker %s (PID %d)", self.model_id, pid)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("Worker PID %d did not die after SIGKILL", pid)
        else:
            logger.info(
                "Worker %s (PID %d) had exited with %s; reaping what it left behind",
                self.model_id,
                pid,
                process.returncode,
            )

        # ``process.wait`` only reaps the launcher. The engine's children
        # (vLLM's EngineCore, SGLang's scheduler) outlive it while they tear
        # down their CUDA contexts, and a replacement started on the same GPUs
        # inside that window fails its startup memory check against the old
        # allocation. Wait for the group to empty, then for the memory.
        stragglers = wait_for_process_group_exit(pgid, timeout=40.0, kill_after=20.0)
        if stragglers:
            logger.error("Worker %s: PIDs %s survived SIGKILL", self.model_id, stragglers)
        if self._gpu_mem_baseline is not None:
            # A region another process still maps (a decode holding a dead
            # prefill's KV) never frees, so the wait is a bounded courtesy.
            waited = time.monotonic()
            held = wait_for_gpu_memory_release(self.gpu_ids, self._gpu_mem_baseline, timeout=30.0)
            if held:
                # The wait gives up once the figure stops moving, usually well
                # inside the 30 s cap; report what was actually waited, and
                # the compute processes nvidia-smi lists on those GPUs. An empty
                # list is not evidence of anything from inside a container (see
                # gpu_compute_apps).
                logger.warning(
                    "Worker %s: GPU memory still held %.0fs after stop: used %s MiB, "
                    "baseline %s MiB, compute processes nvidia-smi lists there: %s",
                    self.model_id,
                    time.monotonic() - waited,
                    held,
                    self._gpu_mem_baseline,
                    gpu_compute_apps(sorted(held)),
                )

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _release_ports(self) -> None:
        from .process_utils import release_port

        release_port(self.port)
        if self.rl_control_port is not None:
            release_port(self.rl_control_port)
            self.rl_control_port = None
        if self.bootstrap_port is not None:
            release_port(self.bootstrap_port)
        if self.nixl_port is not None:
            release_port(self.nixl_port)
            self.nixl_port = None
        if self.dist_init_port is not None:
            release_port(self.dist_init_port)
            self.dist_init_port = None
        if self._admin_port_auto and self.admin_port is not None:
            release_port(self.admin_port)
            self.admin_port = None
            self._admin_port_auto = False
        for port in self._moriio_ports:
            release_port(port)

    def kill_engine(self, sig: int = signal.SIGKILL) -> list[int]:
        """Take the engine away from under its launcher.

        Signals the launcher's live direct children, the engine's own
        processes (vLLM's EngineCore, SGLang's scheduler and detokenizer,
        TokenSpeed's ranks), and leaves the launcher to notice, report and
        exit: the fault a device error or the kernel's out-of-memory killer
        deals an engine under live requests, as opposed to the orderly
        :meth:`stop`. Returns the PIDs signalled; empty when there is no
        launcher, it has exited, or the engine runs inside it (the mock
        worker). :meth:`stop` then :meth:`start` bring the worker back.
        """
        if self.process is None or self.process.poll() is not None:
            return []
        signalled: list[int] = []
        for child in live_child_processes(self.process.pid):
            try:
                os.kill(child, sig)
            except ProcessLookupError:
                continue
            signalled.append(child)
        logger.info(
            "Worker %s (PID %d): signal %d sent to its engine processes %s",
            self.model_id,
            self.process.pid,
            sig,
            signalled,
        )
        return signalled

    def is_alive(self) -> bool:
        """Check if the worker process is still running."""
        return self.process is not None and self.process.poll() is None

    def raise_if_exited(self) -> None:
        """End a start-up wait on this worker's exit: the death report (the
        exit status, what the log says, the log's path), raised at once
        instead of the wait running to its bound."""
        if not self.is_alive():
            raise RuntimeError(self._death_report())

    def effective_kv_backend(self) -> str:
        """This worker's KV transfer backend: its own override, else the lane's."""
        if self.kv_backend:
            return self.kv_backend.lower()
        if self.engine == "sglang":
            return sglang_transfer_backend()
        if self.engine == "tokenspeed":
            # TokenSpeed moves KV over Mooncake only; the lane-wide setting
            # exists for the engines that have a choice.
            return "mooncake"
        return vllm_kv_backend()

    def _build_cmd(self) -> list[str]:
        """Build engine-specific launch command using model specs."""
        spec = get_model_spec(self.model_id)
        model_path = spec["model"]
        tp_size = self.tp or spec.get("tp", 1)
        features = spec.get("features", [])

        if self.engine == "sglang":
            if self.mode == ConnectionMode.ZMQ:
                # The launcher validates the complete engine argument list
                # (it refuses a DP launch this wire does not carry yet), so
                # the extras go through it instead of onto its output.
                return self._build_sglang_zmq_cmd(
                    model_path, tp_size, spec, list(self.extra_engine_args or [])
                )
            cmd = self._build_sglang_cmd(model_path, tp_size, features, spec)
        elif self.engine == "vllm":
            if self.mode == ConnectionMode.ZMQ:
                cmd = self._build_vllm_zmq_cmd(model_path, tp_size, spec)
            elif self.mode == ConnectionMode.GRPC:
                cmd = self._build_vllm_grpc_cmd(model_path, tp_size, spec)
            else:
                cmd = self._build_vllm_http_cmd(model_path, tp_size, spec)
        elif self.engine == "trtllm":
            cmd = self._build_trtllm_cmd(model_path, tp_size, spec)
        elif self.engine == "mlx":
            cmd = self._build_mlx_cmd(model_path, spec)
        elif self.engine == "mock":
            cmd = self._build_mock_cmd(model_path, spec)
        elif self.engine == "tokenspeed":
            if self.mode == ConnectionMode.ZMQ:
                cmd = self._build_tokenspeed_zmq_cmd(model_path, tp_size, spec)
            elif self.mode == ConnectionMode.GRPC:
                cmd = self._build_tokenspeed_grpc_cmd(model_path, tp_size, spec)
            else:
                raise ValueError(
                    "TokenSpeed e2e workers only support gRPC or ZMQ mode; "
                    "HTTP mode would go through the existing OpenAI frontend."
                )
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

        if self.extra_engine_args:
            cmd.extend(self.extra_engine_args)
        return cmd

    def _build_sglang_cmd(
        self, model_path: str, tp_size: int, features: list[str], spec: dict
    ) -> list[str]:
        """Build SGLang launch command."""
        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            DEFAULT_HOST,
            "--port",
            str(self.port),
            "--tp-size",
            str(tp_size),
            "--log-level",
            "warning",
        ]

        if self.mode == ConnectionMode.GRPC:
            cmd.append("--grpc-mode")

        if "embedding" in features:
            cmd.append("--is-embedding")

        # PD disaggregation arguments
        if self.worker_type == WorkerType.PREFILL:
            cmd.extend(["--disaggregation-mode", "prefill"])
            cmd.extend(["--disaggregation-transfer-backend", self.effective_kv_backend()])
            if self.bootstrap_port:
                cmd.extend(["--disaggregation-bootstrap-port", str(self.bootstrap_port)])
            if self.ib_device:
                cmd.extend(["--disaggregation-ib-device", self.ib_device])
        elif self.worker_type == WorkerType.DECODE:
            cmd.extend(["--disaggregation-mode", "decode"])
            cmd.extend(["--disaggregation-transfer-backend", self.effective_kv_backend()])
            cmd.extend(["--base-gpu-id", "0"])
            if self.ib_device:
                cmd.extend(["--disaggregation-ib-device", self.ib_device])

        # Additional SGLang args from model spec (e.g., --context-length)
        sglang_args = spec.get("sglang_args", [])
        if sglang_args:
            cmd.extend(sglang_args)

        return cmd

    def _build_vllm_zmq_cmd(self, model_path: str, tp_size: int, spec: dict) -> list[str]:
        """Build the headless vLLM EngineCore command for the ZMQ direct backend.

        Delegates to the ``smg serve`` launcher so the engine flags and the
        FNV-1a handshake port stay identical to the production launch path.
        """
        from smg.serve import VllmWorkerLauncher

        args = argparse.Namespace(
            connection_mode="zmq", model=model_path, tensor_parallel_size=tp_size
        )
        backend_args = list(spec.get("vllm_args", []))
        # Grouped lane: an engine-level dp flag makes the launcher start that
        # many engines on this worker's socket set (see get_zmq_engine_count).
        engine_count = get_zmq_engine_count()
        if engine_count > 1:
            backend_args += ["--data-parallel-size", str(engine_count)]
        return VllmWorkerLauncher().build_command(args, backend_args, DEFAULT_HOST, self.port)

    def _build_vllm_grpc_cmd(self, model_path: str, tp_size: int, spec: dict) -> list[str]:
        """Build vLLM gRPC server command."""
        return self._build_vllm_base_cmd("vllm.entrypoints.grpc_server", model_path, tp_size, spec)

    def _build_vllm_http_cmd(self, model_path: str, tp_size: int, spec: dict) -> list[str]:
        """Build vLLM HTTP (OpenAI-compatible) server command."""
        return self._build_vllm_base_cmd(
            "vllm.entrypoints.openai.api_server", model_path, tp_size, spec
        )

    def _build_vllm_base_cmd(
        self, entrypoint: str, model_path: str, tp_size: int, spec: dict
    ) -> list[str]:
        """Build shared vLLM command base."""
        cmd = [
            "python3",
            "-m",
            entrypoint,
            "--model",
            model_path,
            "--host",
            DEFAULT_HOST,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(tp_size),
            "--gpu-memory-utilization",
            "0.9",
        ]

        # PD disaggregation: KV transfer roles (backend via E2E_VLLM_KV_BACKEND)
        if self.worker_type in (WorkerType.PREFILL, WorkerType.DECODE):
            kv_role = "kv_producer" if self.worker_type == WorkerType.PREFILL else "kv_consumer"
            backend = self.effective_kv_backend()
            config: dict[str, Any]
            if backend == "mooncake":
                config = {"kv_connector": "MooncakeConnector", "kv_role": kv_role}
            elif backend == "moriio":
                config = {
                    "kv_connector": "MoRIIOConnector",
                    "kv_role": kv_role,
                    "kv_connector_extra_config": self._moriio_extra_config(tp_size),
                }
            else:
                config = {
                    "kv_connector": "NixlConnector",
                    "kv_role": kv_role,
                    # A decode keeps a dead prefill's KV region mapped until it
                    # forgets that peer, and the connector forgets idle peers
                    # only after engine_ttl seconds (an hour by default). A
                    # prefill restarted on the same GPU inside that window
                    # fails its startup free-memory check: on 0.29.0, 53 GiB
                    # of the dead worker stayed resident with the UCX IPC
                    # cache already off (see _build_env). Forget idle peers
                    # after a few seconds instead; the next request to one
                    # re-runs the handshake, which is cheap at this scale.
                    "kv_connector_extra_config": {"engine_ttl": 3},
                }
            cmd.extend(["--kv-transfer-config", json.dumps(config)])

        extra = spec.get("vllm_args", [])
        all_args = [*extra, *(self.extra_engine_args or [])]
        if not any(
            arg.split("=", 1)[0].replace("_", "-") == "--served-model-name" for arg in all_args
        ):
            cmd.extend(["--served-model-name", model_path])
        if extra:
            cmd.extend(extra)
        return cmd

    def _moriio_extra_config(self, tp_size: int) -> dict:
        """MoRIIOConnector settings: this worker's address and ports, the lane's mode."""
        if self.moriio_handshake_port is None:
            self.moriio_handshake_port = get_open_port_block(1)
            self.moriio_notify_port = get_open_port_block(tp_size)
            self._moriio_ports = [
                self.moriio_handshake_port,
                *range(self.moriio_notify_port, self.moriio_notify_port + tp_size),
            ]
        else:
            for port in self._moriio_ports:
                reserve_port(port)
        return {
            "host_ip": moriio_host(),
            "http_port": str(self.port),
            "handshake_port": str(self.moriio_handshake_port),
            "notify_port": str(self.moriio_notify_port),
            "read_mode": vllm_moriio_mode() == "read",
            "backend": vllm_moriio_backend(),
        }

    def moriio_registration(self) -> dict:
        """``POST /workers`` fields that let SMG's HTTP PD router hand KV to this worker.

        A worker registered by URL alone carries no kv_connector, and SMG
        assumes vLLM's default MoRI-IO ports unless the labels give others.
        """
        tp = self.tp or get_model_spec(self.model_id).get("tp", 1)
        prefill = self.worker_type == WorkerType.PREFILL
        return {
            "runtime_type": "vllm",
            "connection_mode": "http",
            "kv_connector": "MoRIIOConnector",
            "kv_role": "kv_producer" if prefill else "kv_consumer",
            "labels": {
                "moriio_mode": vllm_moriio_mode(),
                "moriio_host": moriio_host(),
                "moriio_handshake_port": str(self.moriio_handshake_port),
                "moriio_notify_port": str(self.moriio_notify_port),
                "tp_size": str(tp),
            },
        }

    def _build_mock_cmd(self, model_path: str, spec: dict) -> list[str]:
        """The scripted mock worker (``crates/mock_worker``) on this worker's wire.

        It advertises the spec's model and tokenizer, so the gateway tokenizes
        prompts with the model's files and decodes the ids the mock sends
        back; the mock encodes a scripted text with the same files. The
        binary is ``MOCK_WORKER_BIN`` or ``mock-worker`` on the PATH. Its
        admin API, where a test states the answers, binds ``admin_port``.
        """
        if self.admin_port is None:
            self.admin_port = get_open_port()
            self._admin_port_auto = True
        cmd = [
            os.environ.get("MOCK_WORKER_BIN", "mock-worker"),
            "--host",
            DEFAULT_HOST,
            "--model",
            model_path,
            "--tokenizer",
            model_path,
            "--context-length",
            str(spec.get("context_length", 32768)),
            "--admin-port",
            str(self.admin_port),
        ]
        if self.mode == ConnectionMode.GRPC:
            cmd += ["--grpc-base-port", str(self.port), "--grpc-count", "1"]
        elif self.mode == ConnectionMode.ZMQ:
            # The gateway binds the handshake it derives from the worker's
            # ipc:// URL on the IPv4 loopback whatever the lane's host is
            # (the ZMQ adapter's fixed loopback); the rank dials it, as an
            # engine does.
            from smg.serve import _zmq_handshake_port

            handshake = f"tcp://127.0.0.1:{_zmq_handshake_port(self.base_url)}"
            cmd += ["--zmq-handshake", handshake, "--zmq-count", "1"]
        else:
            cmd += ["--http-base-port", str(self.port), "--http-count", "1"]
        return cmd

    def _build_mlx_cmd(self, model_path: str, spec: dict) -> list[str]:
        """Build MLX gRPC server command (Apple Silicon, gRPC-only)."""
        if self.mode != ConnectionMode.GRPC:
            raise ValueError("MLX backend only supports gRPC mode")
        cmd = [
            "python3",
            "-m",
            "smg_grpc_servicer.mlx.server",
            "--model",
            model_path,
            "--host",
            DEFAULT_HOST,
            "--port",
            str(self.port),
        ]
        extra = spec.get("mlx_args", [])
        if extra:
            cmd.extend(extra)
        return cmd

    def _build_sglang_zmq_cmd(
        self,
        model_path: str,
        tp_size: int,
        spec: dict,
        extra_engine_args: list[str] | None = None,
    ) -> list[str]:
        """Build the headless SGLang command for the ZMQ direct backend.

        Delegates to the ``smg serve`` launcher so the engine flags and the
        FNV-1a handshake port stay identical to the production launch path,
        and so the launcher sees every engine argument, extras included.
        """
        from smg.serve import SglangWorkerLauncher

        args = argparse.Namespace(
            connection_mode="zmq", model_path=model_path, tensor_parallel_size=tp_size
        )
        backend_args = list(spec.get("sglang_args", [])) + list(extra_engine_args or [])
        if tp_size > 1:
            backend_args += ["--tp-size", str(tp_size)]
        return SglangWorkerLauncher().build_command(args, backend_args, DEFAULT_HOST, self.port)

    def _build_tokenspeed_zmq_cmd(self, model_path: str, tp_size: int, spec: dict) -> list[str]:
        """Build the headless TokenSpeed command for the ZMQ direct backend.

        Delegates to the ``smg serve`` launcher so the engine flags and the
        FNV-1a handshake port stay identical to the production launch path.
        """
        from smg.serve import TokenspeedWorkerLauncher

        args = argparse.Namespace(
            connection_mode="zmq", model=model_path, tensor_parallel_size=tp_size
        )
        backend_args = list(spec.get("tokenspeed_args", []))
        # Grouped lane: an engine-level dp flag makes the launcher start that
        # many ranks on this worker's socket set (see get_zmq_engine_count).
        engine_count = get_zmq_engine_count()
        if engine_count > 1:
            backend_args += ["--data-parallel-size", str(engine_count)]
        return TokenspeedWorkerLauncher().build_command(args, backend_args, DEFAULT_HOST, self.port)

    def _build_tokenspeed_grpc_cmd(self, model_path: str, tp_size: int, spec: dict) -> list[str]:
        """Build TokenSpeed gRPC server command.

        Launches ``smg_grpc_servicer.tokenspeed`` which wraps TokenSpeed's
        ``AsyncLLM`` behind the dedicated ``tokenspeed.grpc.scheduler``
        service.
        """
        cmd = [
            "python3",
            "-m",
            "smg_grpc_servicer.tokenspeed",
            "--model",
            model_path,
            "--host",
            DEFAULT_HOST,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(tp_size),
            "--log-level",
            "warning",
            # ``tool_choice=required`` / ``tool_choice={function}`` becomes a
            # json_schema constraint on the wire; TokenSpeed only honors that
            # when a grammar backend is configured (default is ``None``).
            "--grammar-backend",
            "xgrammar",
            # Output logprobs default OFF in tokenspeed; flip them on so
            # ``logprobs=True`` requests get real per-token data back.
            "--enable-output-logprobs",
        ]
        # Only the port: without ``--rl-control-host`` the engine binds the
        # control app on its own ``--host`` (the harness address the gateway
        # already dials) and advertises that URL; a wildcard bind would not be
        # advertised at all. ``ts serve`` always enables the control app, so
        # every lane runs the engine as deployed.
        if self.rl_control_port is None:
            self.rl_control_port = get_engine_port()
        cmd.extend(["--rl-control-port", str(self.rl_control_port)])
        if self.worker_type in (WorkerType.ENCODE, WorkerType.PREFILL, WorkerType.DECODE):
            cmd.extend(["--disaggregation-mode", self.worker_type.value])
            if self.bootstrap_port is not None:
                cmd.extend(["--disaggregation-bootstrap-port", str(self.bootstrap_port)])
            cmd.extend(["--disaggregation-transfer-backend", self.effective_kv_backend()])
            if self.dist_init_addr:
                cmd.extend(["--dist-init-addr", self.dist_init_addr])
            if self.worker_type in (WorkerType.PREFILL, WorkerType.DECODE):
                cmd.append("--enable-prefix-caching")
            if self.worker_type == WorkerType.PREFILL:
                cmd.append("--enforce-eager")

        extra = spec.get("tokenspeed_args", [])
        if extra:
            cmd.extend(extra)
        return cmd

    def _build_trtllm_cmd(self, model_path: str, tp_size: int, spec: dict) -> list[str]:
        """Build TensorRT-LLM gRPC server command."""
        # Create config file to enable xgrammar guided decoding
        config_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="trtllm_"
        )
        config = {"guided_decoding_backend": "xgrammar"}
        config.update(spec.get("trtllm_extra_config", {}))
        yaml.dump(config, config_path, default_flow_style=False)
        config_path.close()

        cmd = [
            "python3",
            "-m",
            "tensorrt_llm.commands.serve",
            "serve",
            model_path,
            "--grpc",
            "--host",
            DEFAULT_HOST,
            "--port",
            str(self.port),
            "--backend",
            "pytorch",
            "--tp_size",
            str(tp_size),
            "--extra_llm_api_options",
            config_path.name,
        ]
        extra = spec.get("trtllm_args", [])
        if extra:
            cmd.extend(extra)
        return cmd

    def _build_env(self) -> dict[str, str]:
        """Build environment variables for the worker process."""
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        if self.engine == "vllm":
            # vLLM 0.31.0 runs tensor-parallel all-reduces through FlashInfer,
            # whose auto backend picks the MNNVL kernel wherever NVLink
            # multicast exists, H100 included. FlashInfer 0.7.0 ships its
            # jit-cache as per-architecture wheels, and the Hopper wheel does
            # not carry that kernel (its AOT list builds it for sm100/sm103
            # only, where the 0.6.x single wheel carried it for every arch),
            # so the worker JIT-compiles it with nvcc, which the pods do not
            # have, and dies at the first all-reduce of the startup memory
            # profile. The TRT-LLM backend is the same fused all-reduce
            # integration and is prebuilt for Hopper. An explicit choice wins.
            env.setdefault("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "trtllm")

        # The servicer implementation of a gRPC worker is a flag inside the smg
        # servicer package, read by the engine's own entrypoint (the command
        # stays): per engine, the variable the worker reads and the lane's
        # choice for it. Built here, not at import, so the getters and the hook
        # check are the module's current ones (a test replaces them).
        servicer_impl = {
            "vllm": ("SMG_VLLM_SERVICER_IMPL", get_vllm_servicer_impl),
            "tokenspeed": ("SMG_TOKENSPEED_SERVICER_IMPL", get_tokenspeed_servicer_impl),
            "sglang": ("SMG_SGLANG_SERVICER_IMPL", get_sglang_servicer_impl),
        }
        # The engines whose Rust servicer needs a hook in the engine package: a
        # worker without it would run Python while the lane reports Rust coverage.
        rust_servicer_needs_hook = {"vllm"}
        if self.mode == ConnectionMode.GRPC and self.engine in servicer_impl:
            variable, lane_choice = servicer_impl[self.engine]
            if lane_choice() == "rust":
                if self.engine in rust_servicer_needs_hook:
                    _require_rust_servicer_hook()
                env[variable] = "rust"
            else:
                # The lane setting is authoritative over an inherited value.
                env.pop(variable, None)

        if (
            self.engine == "vllm"
            and self.mode == ConnectionMode.GRPC
            and get_mm_processing() == MM_PROCESSING_WORKER
        ):
            # The worker advertises mm_processor and the gateway, left in auto
            # mode, forwards media references instead of preprocessed tensors.
            env["SMG_VLLM_MM_PROCESSOR"] = get_vllm_mm_processor()

        if self.engine == "tokenspeed" and self.worker_type in (
            WorkerType.ENCODE,
            WorkerType.PREFILL,
            WorkerType.DECODE,
        ):
            env.setdefault("MC_INTRANODE_NVLINK", "1")
            env.setdefault("TOKENSPEED_SKIP_GRPC_WARMUP", "1")
            env.setdefault("NO_PROXY", "*")
            env.setdefault("no_proxy", "*")

        if (
            self.worker_type in (WorkerType.PREFILL, WorkerType.DECODE)
            and self.effective_kv_backend() == "nixl"
        ):
            # NIXL moves KV over UCX, which binds to the mlx5 device closest to
            # the GPU. A host also exposes mlx5 devices for ordinary VM
            # networking, and picking one of those breaks the worker: it either
            # fails to open the device, when that NIC carries no RoCE GIDs, or
            # opens it and dies on the first transfer, when it has GIDs but
            # cannot route RDMA. Only the workers whose GPUs sit nearest such a
            # NIC are affected, so a lane fails or passes by topology. Pin UCX
            # to the fabric devices to take that choice away.
            fabric_devices = detect_rdma_fabric_devices()
            if fabric_devices:
                env.setdefault("UCX_NET_DEVICES", ",".join(fabric_devices))

        if (
            self.engine == "sglang"
            and self.worker_type in (WorkerType.PREFILL, WorkerType.DECODE)
            and self.effective_kv_backend() == "nixl"
        ):
            # SGLang's NIXL transfer engine rides on UCX too; see the vLLM
            # branch below for why the CUDA IPC cache is off.
            env.setdefault("UCX_CUDA_IPC_CACHE", "n")

        # vLLM PD workers need per-worker side-channel ports for their KV backend
        if self.engine == "vllm" and self.worker_type in (WorkerType.PREFILL, WorkerType.DECODE):
            backend = self.effective_kv_backend()
            if backend == "mooncake":
                # The producer's bootstrap server must listen on the port the
                # gateway advertises in remote_bootstrap_addr
                if self.bootstrap_port is not None:
                    env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(self.bootstrap_port)
            elif backend == "moriio":
                # The address the engine advertises; SMG hands the same one to
                # the peer through the worker's moriio_host label.
                env["VLLM_HOST_IP"] = moriio_host()
            else:
                self.nixl_port = get_engine_port()
                env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(self.nixl_port)
                # Single-host NIXL moves KV over CUDA IPC, and UCX caches the
                # imported handles on the reader. A decode worker then keeps a
                # dead prefill's whole KV region mapped (54 GiB stayed
                # allocated 60 s after every prefill process had exited), and
                # the prefill restarted on that GPU fails its free-memory
                # check. Drop the cache so a mapping ends with its transfer.
                # From vLLM 0.29.0 the reader keeps a mapping per peer anyway,
                # so the engine_ttl in _build_vllm_base_cmd carries the rest.
                env.setdefault("UCX_CUDA_IPC_CACHE", "n")

        if self.engine == "trtllm":
            # TRT-LLM bootstraps workers over Open MPI even at TP=1. On CI pods the
            # MPI OOB/BTL can pick the routable pod interface and flakily fail to
            # TCP-connect to a peer ("connect() to <pod-ip>:1025 failed"), so the
            # engine never starts and the health check times out. All ranks are
            # co-located on one node, so pin MPI to loopback. setdefault keeps any
            # explicit override. Applies at TP=1 too (where the flake was seen).
            env.setdefault("OMPI_MCA_oob_tcp_if_include", "lo")
            env.setdefault("OMPI_MCA_btl_tcp_if_include", "lo")
            # Multi-GPU additionally needs NCCL tuning for CI compatibility.
            if len(self.gpu_ids) > 1:
                env["NCCL_DEBUG"] = "WARN"
                env["NCCL_IB_DISABLE"] = "1"
                env["NCCL_SHM_DISABLE"] = "1"
                env["TLLM_DISABLE_ALLREDUCE_AUTOTUNE"] = "1"

        if self.extra_env:
            env.update(self.extra_env)
        return env

    def read_log(self) -> str:
        """The engine's captured output so far; empty when it goes to the terminal."""
        if self.log_path is None:
            return ""
        with open(self.log_path, encoding="utf-8", errors="replace") as log:
            return log.read()

    def _death_report(self) -> str:
        """The failure text of a worker that exited before it reported SERVING:
        who died with what exit status, what its log says went wrong, and the
        log's path relative to the log directory (the one a CI job uploads).

        The log speaks through its first and last exception lines (the first is
        the root cause of a cause chain, the last the final symptom; a line
        logged below error level does not count, a warning that names an
        exception is not the cause), else its last error-level line, else its
        last non-empty line, each bounded to ``LOG_EXCERPT_CHARS``. An empty
        log leaves the exit status alone.
        """
        pid = self.process.pid if self.process else "unknown"
        code = self.process.returncode if self.process else None
        status = f"exit status {code}"
        if code is not None and code < 0:
            try:
                status = f"signal {signal.Signals(-code).name}"
            except ValueError:
                status = f"signal {-code}"
        report = f"Worker {self.model_id} (PID {pid}) died during startup ({status})"
        lines = [line.rstrip() for line in self.read_log().splitlines() if line.strip()]
        if not lines:
            return report
        exceptions = [
            line
            for line in lines
            if (match := _EXCEPTION_LINE.search(line))
            and not _BELOW_ERROR_LEVEL_LINE.search(line, 0, match.start())
        ]
        errors = [line for line in lines if _ERROR_LEVEL_LINE.search(line)]
        if exceptions:
            picked = [exceptions[0]]
            if exceptions[-1] != exceptions[0]:
                picked.append(exceptions[-1])
        else:
            picked = [errors[-1] if errors else lines[-1]]
        said = " ... ".join(f'"{_excerpt(line)}"' for line in picked)
        path = self.log_path or ""
        if self.log_dir:
            path = os.path.relpath(path, self.log_dir)
        return f"{report}; the log says: {said}; log file {path}"

    def _spawn_process(self, cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
        """Spawn the worker subprocess with output routing."""
        show_output = os.environ.get(ENV_SHOW_WORKER_LOGS, "0") == "1"

        stdout_target: int | IO[Any] | None = None
        stderr_target: int | IO[Any] | None = None

        if not show_output:
            safe_name = f"{self.model_id}_{self.engine}_{self.mode.value}_{self.port}".replace(
                "/", "__"
            ).replace(":", "_")
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                log_path = os.path.join(self.log_dir, f"worker-{safe_name}.log")
                # The run's own directory: a relaunch on this port keeps the
                # previous engine's log beside the new one. The temp fallback
                # below is shared across sessions and stays overwrite-only.
                rotate_previous_log(log_path)
            else:
                log_path = os.path.join(tempfile.gettempdir(), f"smg-worker-{safe_name}.log")
            self._log_file = open(log_path, "w", encoding="utf-8")
            self.log_path = log_path
            stdout_target = self._log_file
            stderr_target = subprocess.STDOUT

        try:
            return subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_target,
                stderr=stderr_target,
                start_new_session=True,
            )
        except Exception:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            raise

    def grpc_serving(self, timeout: float = 5.0) -> bool:
        """One health answer of a gRPC worker: ``True`` on SERVING from its
        ``grpc.health.v1`` service, ``False`` on anything else: a NOT_SERVING,
        a refused connection, and an answer without the health service.

        Every engine servicer of this repository hosts the health service, so
        UNIMPLEMENTED is not this worker speaking. An engine binds its port
        only once it has loaded its model, and another process's listener can
        sit on the number until then: an HTTP/2 server's 404 reads as
        UNIMPLEMENTED through grpc, and the ready channel behind it is the
        stranger's, which the nightly PD sweep once certified as a decode
        worker. One engine's server carries no such service, TRT-LLM
        (``ENGINES_WITHOUT_GRPC_HEALTH``; it answers its own proto's
        HealthCheck, the one the gateway probes); for it alone a ready
        channel that answers UNIMPLEMENTED counts.
        """
        try:
            import grpc
            from grpc_health.v1 import health_pb2, health_pb2_grpc
        except ImportError:
            raise RuntimeError(
                "gRPC health check requires grpcio and grpcio-health-checking packages"
            )

        channel = grpc.insecure_channel(host_port(DEFAULT_HOST, self.port))
        try:
            try:
                stub = health_pb2_grpc.HealthStub(channel)
                request = health_pb2.HealthCheckRequest(service="")
                response = stub.Check(request, timeout=timeout)
                return response.status == health_pb2.HealthCheckResponse.SERVING
            except grpc.RpcError as e:
                if (
                    self.engine in ENGINES_WITHOUT_GRPC_HEALTH
                    and hasattr(e, "code")
                    and e.code() == grpc.StatusCode.UNIMPLEMENTED
                ):
                    try:
                        grpc.channel_ready_future(channel).result(timeout=timeout)
                        return True
                    except Exception:
                        return False
                return False
            except Exception:
                return False
        finally:
            channel.close()

    def _wait_grpc_healthy(self, timeout: float) -> None:
        """Wait for a gRPC worker to become healthy."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            self.raise_if_exited()
            if self.grpc_serving():
                return
            time.sleep(HEALTH_CHECK_INTERVAL)

        raise TimeoutError(
            f"gRPC worker {self.model_id} on port {self.port} "
            f"did not become healthy within {timeout}s"
        )


def _require_rust_servicer_hook() -> None:
    """Fail the Rust lane up front when the installed vLLM cannot select it."""
    try:
        from smg_grpc_servicer.vllm.rust import require_upstream_hook
    except ImportError as error:
        raise RuntimeError(
            f"{ENV_VLLM_SERVICER_IMPL}=rust needs the smg-grpc-servicer package"
        ) from error
    require_upstream_hook()


def worker_startup_timeout(model_id: str, timeout: int = DEFAULT_STARTUP_TIMEOUT) -> int:
    """The bound the harness gives a worker of ``model_id`` to start: its spec's
    ``startup_timeout`` (else ``timeout``), raised to the ``E2E_STARTUP_TIMEOUT``
    floor. ``start_workers`` waits this long for the worker's health; a gateway
    that registers the same worker needs the same bound for its own wait, so
    both read it here and a term added to one reaches the other.
    """
    return effective_startup_timeout(get_model_spec(model_id).get("startup_timeout", timeout))


def start_workers(
    model_id: str,
    engine: str | None = None,
    mode: ConnectionMode = ConnectionMode.HTTP,
    count: int = 1,
    worker_type: WorkerType = WorkerType.REGULAR,
    timeout: int = DEFAULT_STARTUP_TIMEOUT,
    log_dir: str | None = None,
    gpu_offset: int = 0,
    wait_ready: bool = True,
    gpus: int | None = None,
    extra_engine_args: list[str] | None = None,
    tp: int | None = None,
    kv_backend: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[Worker]:
    """Start N workers for a model. GPU IDs assigned sequentially.

    Args:
        model_id: Model identifier from MODEL_SPECS.
        engine: Runtime engine ("sglang", "vllm", "trtllm").
                If None, auto-detected from E2E_RUNTIME env var.
        mode: Connection mode (HTTP or GRPC).
        count: Number of workers to start.
        worker_type: Worker specialization (regular, prefill, decode).
        timeout: Shared startup bound for all workers to become healthy.
        log_dir: Directory to store worker log files.
        gpu_offset: Starting GPU index for worker assignment.
        wait_ready: If True (default), spawn all workers before waiting for health.
            If False, spawn processes and return immediately.
        gpus: GPUs per worker; defaults to the model spec's tp (e.g. DP needs dp*tp).
        extra_engine_args: Extra CLI args appended to the engine launch command.
        tp: Tensor-parallel size for these workers; defaults to the model
            spec's tp. Also sizes the GPU slice unless ``gpus`` says otherwise.
        kv_backend: KV transfer backend for PD workers ("nixl" or
            "mooncake"); defaults to the lane's setting.

    Returns:
        List of started Worker instances.
    """
    # The lane's GPU slice: every worker shifts by E2E_GPU_OFFSET.
    gpu_offset += get_gpu_offset()
    if log_dir is None:
        log_dir = os.environ.get("E2E_LOG_DIR")

    if engine is None:
        engine = get_runtime()

    spec = get_model_spec(model_id)
    gpus_per_worker = gpus or tp or spec.get("tp", 1)
    if gpus is None and mode == ConnectionMode.ZMQ:
        # A grouped ZMQ worker launches get_zmq_engine_count() engines, each
        # tp-wide, in one process — size its GPU slice accordingly. vLLM and
        # TokenSpeed launchers both start engine groups; a grouped count on
        # any other runtime would reserve GPUs for engines that never launch
        # and leave the gateway awaiting handshakes that never come.
        zmq_engine_count = get_zmq_engine_count()
        if zmq_engine_count > 1 and engine not in ("vllm", "tokenspeed"):
            raise ValueError(
                f"E2E_ZMQ_ENGINE_COUNT={zmq_engine_count} needs an engine-group "
                f"launcher (vllm or tokenspeed); got engine={engine!r}"
            )
        gpus_per_worker *= zmq_engine_count
    timeout = worker_startup_timeout(model_id, timeout)

    # Detect IB device for PD workers
    has_pd = worker_type in (WorkerType.PREFILL, WorkerType.DECODE)
    ib_device = detect_ib_device() if has_pd else None

    workers: list[Worker] = []

    try:
        for i in range(count):
            gpu_ids = list(range(gpu_offset, gpu_offset + gpus_per_worker))
            gpu_offset += gpus_per_worker
            # The engine binds these once it has loaded its model: numbers the
            # kernel never hands to another process's bind(0) meanwhile.
            port = get_engine_port()
            bootstrap_port = (
                get_engine_port()
                if worker_type in (WorkerType.PREFILL, WorkerType.ENCODE)
                else None
            )
            is_ts_disagg = engine == "tokenspeed" and worker_type in (
                WorkerType.ENCODE,
                WorkerType.PREFILL,
                WorkerType.DECODE,
            )
            dist_init_port = get_engine_port() if is_ts_disagg else None
            dist_init_addr = host_port(DEFAULT_HOST, dist_init_port) if dist_init_port else None

            worker = Worker(
                model_id=model_id,
                engine=engine,
                port=port,
                gpu_ids=gpu_ids,
                mode=mode,
                worker_type=worker_type,
                bootstrap_port=bootstrap_port,
                ib_device=ib_device if has_pd else None,
                dist_init_addr=dist_init_addr,
                dist_init_port=dist_init_port,
                log_dir=log_dir,
                extra_engine_args=extra_engine_args,
                tp=tp,
                kv_backend=kv_backend,
                extra_env=extra_env,
            )

            # Stagger launches to avoid resource contention
            if i > 0 and LAUNCH_STAGGER_DELAY > 0:
                logger.info("Staggering launch by %ds", LAUNCH_STAGGER_DELAY)
                time.sleep(LAUNCH_STAGGER_DELAY)

            workers.append(worker)
            # Each worker has its own GPU slice. Let model loading overlap
            # instead of holding later workers' GPUs idle during this wait.
            worker.start(timeout=timeout, wait_ready=False)
        if wait_ready:
            # Launch staggering must not shorten the last worker's load budget.
            deadline = time.monotonic() + timeout
            for worker in workers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Workers for {model_id} did not become healthy within {timeout}s"
                    )
                worker.wait_ready(max(1, int(remaining)))
    except Exception:
        stop_workers(workers)
        raise

    return workers


def stop_workers(workers: list[Worker]) -> None:
    """Stop all workers in a list.

    Args:
        workers: List of Worker instances to stop.
    """
    for worker in workers:
        try:
            worker.stop()
        except Exception as e:
            logger.warning("Error stopping worker %s: %s", worker.model_id, e)
