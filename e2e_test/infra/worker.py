"""Simple worker lifecycle management. No pooling, no eviction, no sharing."""

from __future__ import annotations

import argparse
import json
import logging
import os
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
    HEALTH_CHECK_INTERVAL,
    LAUNCH_STAGGER_DELAY,
    ConnectionMode,
    WorkerType,
    get_runtime,
    get_zmq_engine_count,
    vllm_kv_backend,
)
from .model_specs import get_model_spec
from .process_utils import detect_ib_device, get_open_port, wait_for_health

logger = logging.getLogger(__name__)


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
    ib_device: str | None = None
    dist_init_addr: str | None = None
    dist_init_port: int | None = None
    log_dir: str | None = None
    extra_engine_args: list[str] | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)
    _log_file: IO[Any] | None = field(default=None, repr=False)

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
            return f"grpc://{DEFAULT_HOST}:{self.port}"
        return f"http://{DEFAULT_HOST}:{self.port}"

    @property
    def worker_url(self) -> str:
        """Alias for base_url for Gateway compatibility."""
        return self.base_url

    @property
    def http_url(self) -> str:
        """HTTP URL (used for health checks even on gRPC workers)."""
        return f"http://{DEFAULT_HOST}:{self.port}"

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

        self.process = self._spawn_process(cmd, env)

        if not wait_ready:
            logger.info(
                "Worker %s spawned at %s (PID %d) — not waiting for health",
                self.model_id,
                self.base_url,
                self.process.pid,
            )
            return

        # Wait for health check
        if self.mode == ConnectionMode.ZMQ:
            # SMG (the router) binds the ZMQ sockets and this engine dials in;
            # there is no worker port to probe. The gateway's readiness gate
            # (wait_for_workers_ready) covers the engine, so just proceed.
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
            wait_for_health(
                self.http_url,
                timeout=timeout,
                check_interval=HEALTH_CHECK_INTERVAL,
            )

        logger.info(
            "Worker %s healthy at %s (PID %d)",
            self.model_id,
            self.base_url,
            self.process.pid,
        )

    def stop(self) -> None:
        """Terminate worker process and all child processes."""
        if self.process is None or self.process.poll() is not None:
            return

        pid = self.process.pid
        logger.info("Stopping worker %s (PID %d)", self.model_id, pid)

        # Kill entire process group (workers run in their own session)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            self.process.terminate()

        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                self.process.kill()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("Worker PID %d did not die after SIGKILL", pid)

        # Clean up log file
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        # Release reserved ports
        from .process_utils import release_port

        release_port(self.port)
        if self.bootstrap_port is not None:
            release_port(self.bootstrap_port)
        if self.nixl_port is not None:
            release_port(self.nixl_port)
            self.nixl_port = None
        if self.dist_init_port is not None:
            release_port(self.dist_init_port)
            self.dist_init_port = None

    def is_alive(self) -> bool:
        """Check if the worker process is still running."""
        return self.process is not None and self.process.poll() is None

    def _build_cmd(self) -> list[str]:
        """Build engine-specific launch command using model specs."""
        spec = get_model_spec(self.model_id)
        model_path = spec["model"]
        tp_size = spec.get("tp", 1)
        features = spec.get("features", [])

        if self.engine == "sglang":
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
            if self.bootstrap_port:
                cmd.extend(["--disaggregation-bootstrap-port", str(self.bootstrap_port)])
            if self.ib_device:
                cmd.extend(["--disaggregation-ib-device", self.ib_device])
        elif self.worker_type == WorkerType.DECODE:
            cmd.extend(["--disaggregation-mode", "decode"])
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
            if vllm_kv_backend() == "mooncake":
                config = {"kv_connector": "MooncakeConnector", "kv_role": kv_role}
            else:
                config = {"kv_connector": "NixlConnector", "kv_role": kv_role}
            cmd.extend(["--kv-transfer-config", json.dumps(config)])

        extra = spec.get("vllm_args", [])
        if extra:
            cmd.extend(extra)
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
        if self.worker_type in (WorkerType.ENCODE, WorkerType.PREFILL, WorkerType.DECODE):
            cmd.extend(["--disaggregation-mode", self.worker_type.value])
            if self.bootstrap_port is not None:
                cmd.extend(["--disaggregation-bootstrap-port", str(self.bootstrap_port)])
            cmd.extend(["--disaggregation-transfer-backend", "mooncake"])
            if self.dist_init_addr:
                cmd.extend(["--dist-init-addr", self.dist_init_addr])
            if self.worker_type in (WorkerType.PREFILL, WorkerType.DECODE):
                cmd.append("--enable-prefix-caching")
            if self.worker_type == WorkerType.PREFILL:
                cmd.append("--enforce-eager")
            cmd.append("--skip-server-warmup")

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

        if self.engine == "tokenspeed" and self.worker_type in (
            WorkerType.ENCODE,
            WorkerType.PREFILL,
            WorkerType.DECODE,
        ):
            env.setdefault("MC_INTRANODE_NVLINK", "1")
            env.setdefault("TOKENSPEED_SKIP_GRPC_WARMUP", "1")
            env.setdefault("NO_PROXY", "*")
            env.setdefault("no_proxy", "*")

        # vLLM PD workers need per-worker side-channel ports for their KV backend
        if self.engine == "vllm" and self.worker_type in (WorkerType.PREFILL, WorkerType.DECODE):
            if vllm_kv_backend() == "mooncake":
                # The producer's bootstrap server must listen on the port the
                # gateway advertises in remote_bootstrap_addr
                if self.bootstrap_port is not None:
                    env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(self.bootstrap_port)
            else:
                self.nixl_port = get_open_port()
                env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(self.nixl_port)

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

        return env

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
            else:
                log_path = os.path.join(tempfile.gettempdir(), f"smg-worker-{safe_name}.log")
            self._log_file = open(log_path, "w", encoding="utf-8")
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

    def _wait_grpc_healthy(self, timeout: float) -> None:
        """Wait for a gRPC worker to become healthy."""
        try:
            import grpc
            from grpc_health.v1 import health_pb2, health_pb2_grpc
        except ImportError:
            raise RuntimeError(
                "gRPC health check requires grpcio and grpcio-health-checking packages"
            )

        start = time.perf_counter()
        channel = grpc.insecure_channel(f"{DEFAULT_HOST}:{self.port}")
        try:
            while time.perf_counter() - start < timeout:
                if not self.is_alive():
                    pid = self.process.pid if self.process else "unknown"
                    raise RuntimeError(f"Worker {self.model_id} (PID {pid}) died during startup")
                try:
                    stub = health_pb2_grpc.HealthStub(channel)
                    request = health_pb2.HealthCheckRequest(service="")
                    response = stub.Check(request, timeout=5.0)
                    if response.status == health_pb2.HealthCheckResponse.SERVING:
                        return
                except grpc.RpcError as e:
                    # UNIMPLEMENTED means server is up but doesn't have health service;
                    # fall back to channel connectivity check
                    if hasattr(e, "code") and e.code() == grpc.StatusCode.UNIMPLEMENTED:
                        try:
                            grpc.channel_ready_future(channel).result(timeout=5.0)
                            return
                        except Exception:
                            pass
                except Exception:
                    pass

                time.sleep(HEALTH_CHECK_INTERVAL)
        finally:
            channel.close()

        raise TimeoutError(
            f"gRPC worker {self.model_id} on port {self.port} "
            f"did not become healthy within {timeout}s"
        )


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
) -> list[Worker]:
    """Start N workers for a model. GPU IDs assigned sequentially.

    Args:
        model_id: Model identifier from MODEL_SPECS.
        engine: Runtime engine ("sglang", "vllm", "trtllm").
                If None, auto-detected from E2E_RUNTIME env var.
        mode: Connection mode (HTTP or GRPC).
        count: Number of workers to start.
        worker_type: Worker specialization (regular, prefill, decode).
        timeout: Seconds to wait for each worker to become healthy.
        log_dir: Directory to store worker log files.
        gpu_offset: Starting GPU index for worker assignment.
        wait_ready: If True (default), block until each worker is healthy.
            If False, spawn processes and return immediately.
        gpus: GPUs per worker; defaults to the model spec's tp (e.g. DP needs dp*tp).
        extra_engine_args: Extra CLI args appended to the engine launch command.

    Returns:
        List of started Worker instances.
    """
    if log_dir is None:
        log_dir = os.environ.get("E2E_LOG_DIR")

    if engine is None:
        engine = get_runtime()

    spec = get_model_spec(model_id)
    gpus_per_worker = gpus or spec.get("tp", 1)
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
    timeout = spec.get("startup_timeout", timeout)

    # Detect IB device for PD workers
    has_pd = worker_type in (WorkerType.PREFILL, WorkerType.DECODE)
    ib_device = detect_ib_device() if has_pd else None

    workers: list[Worker] = []

    try:
        for i in range(count):
            gpu_ids = list(range(gpu_offset, gpu_offset + gpus_per_worker))
            gpu_offset += gpus_per_worker
            port = get_open_port()
            bootstrap_port = (
                get_open_port() if worker_type in (WorkerType.PREFILL, WorkerType.ENCODE) else None
            )
            is_ts_disagg = engine == "tokenspeed" and worker_type in (
                WorkerType.ENCODE,
                WorkerType.PREFILL,
                WorkerType.DECODE,
            )
            dist_init_port = get_open_port() if is_ts_disagg else None
            dist_init_addr = f"{DEFAULT_HOST}:{dist_init_port}" if dist_init_port else None

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
            )

            # Stagger launches to avoid resource contention
            if i > 0 and LAUNCH_STAGGER_DELAY > 0:
                logger.info("Staggering launch by %ds", LAUNCH_STAGGER_DELAY)
                time.sleep(LAUNCH_STAGGER_DELAY)

            workers.append(worker)
            worker.start(timeout=timeout, wait_ready=wait_ready)
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
