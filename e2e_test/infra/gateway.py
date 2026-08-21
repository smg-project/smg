"""Gateway class for managing Shepherd Model Gateway router instances."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .constants import (
    DEFAULT_HOST,
    DEFAULT_ROUTER_TIMEOUT,
    ENV_SHOW_ROUTER_LOGS,
    get_zmq_engine_count,
)
from .process_utils import (
    get_open_port,
    kill_process_tree,
    release_port,
    wait_for_health,
    wait_for_workers_ready,
)

if TYPE_CHECKING:
    from .worker import Worker

logger = logging.getLogger(__name__)


def build_epd_mode_args(
    encode_workers: list[Worker],
    prefill_workers: list[Worker],
    decode_workers: list[Worker],
    encode_policy: str = "consistent_hashing",
) -> list[str]:
    """Build the ``smg.launch_router`` args for EPD disaggregation.

    ``--encode`` / ``--prefill`` are repeatable ``URL [BOOTSTRAP_PORT]`` flags;
    ``--decode`` is a repeatable ``URL`` flag. Multimodal tensors ride inline
    (SMG -> encode); encode -> prefill embeddings and prefill -> decode KV move
    over Mooncake independently of this flag.
    """
    args = ["--epd-disaggregation"]
    for en in encode_workers:
        args += ["--encode", en.base_url]
        if en.bootstrap_port is not None:
            args.append(str(en.bootstrap_port))
    for pf in prefill_workers:
        args += ["--prefill", pf.base_url]
        if pf.bootstrap_port is not None:
            args.append(str(pf.bootstrap_port))
    for dc in decode_workers:
        args += ["--decode", dc.base_url]
    if encode_policy:
        args += ["--encode-policy", encode_policy]
    args += ["--multimodal-tensor-transport", "inline"]
    return args


@dataclass
class WorkerInfo:
    """Information about a worker connected to the gateway."""

    id: str
    url: str
    model: str | None = None
    status: str = "unknown"
    pending_requests: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Gateway:
    """Manages a Shepherd Model Gateway router instance.

    Four startup modes:
    - Regular: start(worker_urls=[...], model_path="...")
    - PD: start(prefill_workers=[...], decode_workers=[...])
    - IGW: start(igw_mode=True), then add_worker(url) dynamically
    - Cloud: start(cloud_backend="openai"|"xai"|"anthropic")
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int | None = None,
        prometheus_port: int | None = None,
    ):
        self.host = host
        self._port_auto_allocated = port is None
        self._prometheus_port_auto_allocated = prometheus_port is None
        self.port = port or get_open_port()
        self.prometheus_port = prometheus_port or get_open_port()
        self.base_url = f"http://{self.host}:{self.port}"
        self.metrics_url = f"http://{self.host}:{self.prometheus_port}"

        self.process: subprocess.Popen | None = None
        self.model_path: str | None = None
        self.policy: str = "round_robin"
        self.log_level: str = "warn"
        self.log_dir: str | None = None
        self.pd_mode: bool = False
        self.igw_mode: bool = False
        self.cloud_mode: bool = False
        self.cloud_backend: str | None = None
        self._started: bool = False
        self._env: dict[str, str] | None = None

    @property
    def is_running(self) -> bool:
        """Check if the gateway process is running."""
        return self.process is not None and self.process.poll() is None

    def start(
        self,
        *,
        worker_urls: list[str] | None = None,
        model_path: str | None = None,
        prefill_workers: list[Worker] | None = None,
        decode_workers: list[Worker] | None = None,
        encode_workers: list[Worker] | None = None,
        encode_policy: str = "consistent_hashing",
        igw_mode: bool = False,
        cloud_backend: str | None = None,
        history_backend: str = "memory",
        backend: str | None = None,
        policy: str = "round_robin",
        timeout: float = DEFAULT_ROUTER_TIMEOUT,
        show_output: bool | None = None,
        extra_args: list[str] | None = None,
        log_level: str | None = None,
        log_dir: str | None = None,
    ) -> None:
        """Start the gateway in exactly one mode (regular, PD, IGW, or cloud)."""
        if self._started:
            raise RuntimeError("Gateway already started")

        is_epd_mode = encode_workers is not None
        is_pd_mode = (prefill_workers is not None or decode_workers is not None) and not is_epd_mode
        is_regular_mode = worker_urls is not None
        is_igw_mode = igw_mode
        is_cloud_mode = cloud_backend is not None

        modes_specified = sum(
            [is_epd_mode, is_pd_mode, is_regular_mode, is_igw_mode, is_cloud_mode]
        )
        if modes_specified != 1:
            raise ValueError(
                "Specify exactly one mode: worker_urls, encode/prefill/decode_workers "
                "(EPD), prefill/decode_workers (PD), igw_mode=True, or cloud_backend"
            )

        if show_output is None:
            show_output = os.environ.get(ENV_SHOW_ROUTER_LOGS, "0") == "1"

        self.policy = policy
        if log_level:
            self.log_level = log_level
        if log_dir:
            self.log_dir = log_dir

        if is_igw_mode:
            self.pd_mode = False
            self.igw_mode = True
            self._launch(
                mode_args=["--enable-igw"],
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
                log_msg="IGW gateway (no workers)",
            )
        elif is_epd_mode:
            self.pd_mode = True
            self.igw_mode = False
            encodes = encode_workers or []
            prefills = prefill_workers or []
            decodes = decode_workers or []
            mode_args = build_epd_mode_args(encodes, prefills, decodes, encode_policy)
            self._launch(
                mode_args=mode_args,
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
                log_msg=(
                    f"EPD gateway ({len(encodes)} encode, "
                    f"{len(prefills)} prefill, {len(decodes)} decode)"
                ),
            )
        elif is_pd_mode:
            self.pd_mode = True
            self.igw_mode = False
            prefills = prefill_workers or []
            decodes = decode_workers or []

            mode_args = ["--pd-disaggregation"]
            for pf in prefills:
                if pf.bootstrap_port is not None:
                    mode_args += ["--prefill", pf.base_url, str(pf.bootstrap_port)]
                else:
                    mode_args += ["--prefill", pf.worker_url]
            for dc in decodes:
                mode_args += ["--decode", dc.worker_url]

            self._launch(
                mode_args=mode_args,
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
                log_msg=f"PD gateway ({len(prefills)} prefill, {len(decodes)} decode)",
            )
        elif is_cloud_mode:
            assert cloud_backend is not None
            self.pd_mode = False
            self.igw_mode = False
            self.cloud_mode = True
            self.cloud_backend = cloud_backend
            mode_args = self._build_cloud_args(cloud_backend, history_backend)
            self._launch(
                mode_args=mode_args,
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
                num_workers=1,
                log_msg=f"{cloud_backend} cloud gateway",
            )
        else:
            if not model_path:
                raise ValueError("model_path is required for regular mode")
            if not worker_urls:
                raise ValueError("worker_urls must be non-empty for regular mode")
            self.model_path = model_path
            self.pd_mode = False
            self.igw_mode = False
            mode_args = ["--model-path", model_path, "--worker-urls", *worker_urls]
            # ZMQ workers share one wire across engine runtimes, so the router
            # cannot probe the backend from the ipc:// URL — pin it explicitly.
            if backend is not None:
                mode_args += ["--backend", backend]
            # Grouped ZMQ lane: the handshake must await every engine the
            # worker launched (see get_zmq_engine_count).
            engine_count = get_zmq_engine_count()
            if engine_count > 1:
                mode_args += ["--zmq-engine-count", str(engine_count)]
            self._launch(
                mode_args=mode_args,
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
                num_workers=len(worker_urls),
                log_msg=f"gateway with {len(worker_urls)} worker(s)",
            )

    def _launch(
        self,
        mode_args: list[str],
        timeout: float,
        show_output: bool,
        extra_args: list[str] | None,
        num_workers: int | None = None,
        log_msg: str = "",
    ) -> None:
        """Launch the gateway process and wait for it to become ready."""
        cmd = self._build_base_cmd()
        cmd.extend(mode_args)

        if extra_args:
            cmd.extend(extra_args)

        logger.info("Starting %s on port %d", log_msg or "gateway", self.port)
        logger.debug("Gateway command: %s", " ".join(cmd))

        stdout_target = None if show_output else subprocess.DEVNULL
        stderr_target = None if show_output else subprocess.DEVNULL

        self.process = subprocess.Popen(
            cmd,
            env=self._env,
            stdout=stdout_target,
            stderr=stderr_target,
            start_new_session=True,
        )

        try:
            if num_workers is not None:
                wait_for_workers_ready(self.base_url, num_workers, timeout=timeout)
            else:
                wait_for_health(self.base_url, timeout=timeout)
        except TimeoutError:
            self.shutdown()
            raise

        self._started = True
        logger.info("Gateway ready at %s", self.base_url)

    def shutdown(self) -> None:
        """Shutdown the gateway and release auto-allocated ports."""
        if self.process is not None:
            logger.info("Shutting down gateway (PID %d)", self.process.pid)
            kill_process_tree(self.process.pid)
            self.process = None

        if self._port_auto_allocated:
            release_port(self.port)
        if self._prometheus_port_auto_allocated:
            release_port(self.prometheus_port)

        self._started = False

    def _build_base_cmd(self) -> list[str]:
        """Build the base command for launching the router."""
        cmd = [
            "python3",
            "-m",
            "smg.launch_router",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--prometheus-port",
            str(self.prometheus_port),
            "--prometheus-host",
            self.host,
            "--policy",
            self.policy,
            "--log-level",
            self.log_level,
        ]
        if self.log_dir:
            cmd.extend(["--log-dir", self.log_dir])
        return cmd

    def _build_cloud_args(self, cloud_backend: str, history_backend: str) -> list[str]:
        """Build CLI args and env for cloud mode."""
        cloud_configs = {
            "openai": ("https://api.openai.com", "OPENAI_API_KEY", "openai"),
            "xai": ("https://api.x.ai", "XAI_API_KEY", "openai"),
            "anthropic": ("https://api.anthropic.com", "ANTHROPIC_API_KEY", "anthropic"),
        }
        if cloud_backend not in cloud_configs:
            raise ValueError(f"Unsupported cloud backend: {cloud_backend}")

        worker_url, api_key_env, backend_type = cloud_configs[cloud_backend]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} environment variable required")

        self._env = os.environ.copy()
        self._env[api_key_env] = api_key

        actual_history_backend = history_backend
        if history_backend == "oracle-custom":
            actual_history_backend = "oracle"
            flyway_user = os.environ.get("ATP_FLYWAY_USER", "")
            flyway_password = os.environ.get("ATP_FLYWAY_PASSWORD", "")
            flyway_dsn = os.environ.get("ATP_FLYWAY_DSN", "")
            if not all([flyway_user, flyway_password, flyway_dsn]):
                raise ValueError(
                    "ATP_FLYWAY_USER, ATP_FLYWAY_PASSWORD, and ATP_FLYWAY_DSN "
                    "environment variables required for oracle-custom backend"
                )
            self._env["ATP_USER"] = flyway_user
            self._env["ATP_PASSWORD"] = flyway_password
            self._env["ATP_DSN"] = flyway_dsn

        mode_args = [
            "--backend",
            backend_type,
            "--worker-urls",
            worker_url,
            "--history-backend",
            actual_history_backend,
            "--disable-health-check",
        ]

        if history_backend == "oracle-custom":
            schema_config_path = (
                Path(__file__).resolve().parents[2]
                / "scripts"
                / "oracle_flyway"
                / "schema-config.yaml"
            )
            mode_args.extend(["--schema-config", str(schema_config_path)])

        return mode_args

    def health(self, timeout: float = 5.0) -> bool:
        """Check gateway health. Returns True if healthy."""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=timeout)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def _worker_from_api_response(self, w: dict) -> WorkerInfo:
        """Convert API response dict to WorkerInfo."""
        status = "healthy" if w.get("is_healthy", False) else "unhealthy"
        return WorkerInfo(
            id=w.get("id", ""),
            url=w.get("url", ""),
            model=w.get("model_id"),
            status=status,
            pending_requests=w.get("load", 0),
            metadata={
                "worker_type": w.get("worker_type"),
                "connection_mode": w.get("connection_mode"),
                "priority": w.get("priority"),
                "cost": w.get("cost"),
            },
        )

    def list_workers(self, timeout: float = 5.0, strict: bool = False) -> list[WorkerInfo]:
        """List all workers connected to the gateway.

        With ``strict=True``, request failures and non-200 responses raise
        instead of degrading to ``[]`` — required when an empty list is the
        assertion target (e.g. "worker was removed"), where a swallowed error
        would pass vacuously.
        """
        try:
            resp = httpx.get(f"{self.base_url}/workers", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return [self._worker_from_api_response(w) for w in data.get("workers", [])]
            if strict:
                raise RuntimeError(f"GET /workers returned {resp.status_code}: {resp.text}")
            return []
        except (httpx.RequestError, httpx.TimeoutException):
            if strict:
                raise
            return []

    def add_worker(
        self,
        worker_url: str,
        timeout: float = 10.0,
        wait_ready: bool = True,
        ready_timeout: float = 60.0,
        labels: dict[str, str] | None = None,
    ) -> tuple[bool, str | None]:
        """Add a worker to the gateway. Returns (success, worker_id or error).

        ``labels`` are attached to the worker's ``WorkerSpec`` (e.g.
        ``{"realtime": "true"}`` to make it eligible for realtime routing).
        """
        body: dict = {"url": worker_url}
        if labels:
            body["labels"] = labels
        try:
            resp = httpx.post(
                f"{self.base_url}/workers",
                json=body,
                timeout=timeout,
            )
            if resp.status_code in (200, 202):
                data = resp.json()
                worker_id = data.get("worker_id")

                if wait_ready and worker_id:
                    start = time.perf_counter()
                    while time.perf_counter() - start < ready_timeout:
                        workers = self.list_workers()
                        for w in workers:
                            if w.id == worker_id:
                                return True, worker_id
                        time.sleep(1.0)
                    return False, f"Worker {worker_id} not ready within {ready_timeout}s"

                return True, worker_id
            return False, resp.text
        except (httpx.RequestError, httpx.TimeoutException) as e:
            return False, str(e)

    def remove_worker(self, worker_url: str, timeout: float = 10.0) -> tuple[bool, str]:
        """Remove a worker from the gateway by URL. Returns (success, message)."""
        workers = self.list_workers(timeout=timeout)
        worker_id = next((w.id for w in workers if w.url == worker_url), None)

        if not worker_id:
            return False, f"Worker with URL {worker_url} not found"

        try:
            resp = httpx.delete(
                f"{self.base_url}/workers/{worker_id}",
                timeout=timeout,
            )
            # 200 = removed synchronously; 202 = removal accepted and queued
            # for background processing. Either means the request succeeded —
            # callers that need completion poll list_workers for absence.
            if resp.status_code in (200, 202):
                return True, resp.text
            return False, resp.text
        except (httpx.RequestError, httpx.TimeoutException) as e:
            return False, str(e)

    def list_models(self, timeout: float = 5.0) -> list[dict]:
        """List available models (OpenAI-compatible)."""
        try:
            resp = httpx.get(f"{self.base_url}/v1/models", timeout=timeout)
            if resp.status_code == 200:
                return resp.json().get("data", [])
            return []
        except (httpx.RequestError, httpx.TimeoutException):
            return []

    def __enter__(self) -> Gateway:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


def launch_cloud_gateway(
    runtime: str,
    *,
    history_backend: str = "memory",
    extra_args: list[str] | None = None,
    timeout: float = 60,
    show_output: bool | None = None,
    max_attempts: int = 3,
) -> Gateway:
    """Launch gateway with a cloud API runtime (openai, xai, anthropic).

    Retries up to max_attempts because the AddWorker workflow can fail
    intermittently due to a race condition in the workflow engine's
    parallel step context handling.
    """
    from .model_specs import THIRD_PARTY_MODELS

    if runtime not in THIRD_PARTY_MODELS:
        raise ValueError(
            f"Unknown cloud runtime: {runtime}. Available: {list(THIRD_PARTY_MODELS.keys())}"
        )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        gateway = Gateway()
        try:
            gateway.start(
                cloud_backend=runtime,
                history_backend=history_backend,
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
            )
            return gateway
        except TimeoutError as e:
            last_error = e
            logger.warning(
                "Cloud gateway startup attempt %d/%d timed out: %s",
                attempt,
                max_attempts,
                e,
            )
            gateway.shutdown()
    raise last_error  # type: ignore[misc]
