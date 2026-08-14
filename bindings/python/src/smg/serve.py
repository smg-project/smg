"""
Serve command: two-pass CLI argument parsing with lazy backend import.

Launches backend worker(s) + gateway router via a single `smg serve` command.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import time
from abc import ABC, abstractmethod

from smg.launch_router import launch_router
from smg.router_args import RouterArgs

logger = logging.getLogger("smg.serve")

# Directory for the per-worker ipc:// sockets when connection_mode == "zmq".
# SMG derives its data-plane socket paths and the tcp handshake port from the
# ipc:// worker URL, so the engine and the router only need to agree on this URL.
# Per-user directory for the ZMQ ipc:// sockets. Scoped to the current uid and
# created 0700 (single-owner) so a shared /tmp cannot leak another user's
# sockets into this router. Override with SMG_ZMQ_SOCKET_DIR.
_ZMQ_SOCKET_DIR = os.environ.get("SMG_ZMQ_SOCKET_DIR", f"/tmp/smg-zmq-{os.getuid()}")


def _zmq_ipc_url(port: int) -> str:
    """Per-worker ipc:// URL. The port keeps it unique across DP-free workers."""
    return f"ipc://{_ZMQ_SOCKET_DIR}/engine-{port}"


def _backend_arg_int(backend_args: list[str], flag: str, default: int) -> int:
    """Read an integer flag from raw backend args (``--flag N`` or ``--flag=N``).

    An unparsable value falls back to the default with a warning — the
    launcher owns this flag downstream, so silence would quietly discard the
    operator's intent.
    """
    for i, arg in enumerate(backend_args):
        value = None
        if arg == flag and i + 1 < len(backend_args):
            value = backend_args[i + 1]
        elif arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
        if value is not None:
            try:
                return int(value)
            except ValueError:
                logger.warning(
                    "Ignoring non-integer value %r for %s; using %d", value, flag, default
                )
                return default
    return default


# The band `derive_handshake_port` folds every worker's handshake port into.
# Other launcher-derived tcp ports must stay out of it: a port in this band can
# collide with SOME worker's handshake listener for the right ipc path.
_ZMQ_HANDSHAKE_PORT_BASE = 20000
_ZMQ_HANDSHAKE_PORT_SPAN = 10000


def _zmq_handshake_port(ipc_url: str) -> int:
    """The tcp handshake port SMG derives from an ipc:// URL.

    Mirrors `derive_handshake_port` in model_gateway/src/worker/worker.rs exactly
    (FNV-1a over the path after `ipc://`, mapped into 20000..=29999) so the engine
    can be launched with the matching `--data-parallel-rpc-port`.
    """
    path = ipc_url[len("ipc://") :] if ipc_url.startswith("ipc://") else ipc_url
    h = 0xCBF29CE484222325
    for b in path.encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return _ZMQ_HANDSHAKE_PORT_BASE + (h % _ZMQ_HANDSHAKE_PORT_SPAN)


def _reject_handshake_port_collisions(ports: list[int]) -> None:
    """Fail before launch if two workers derive the same ZMQ handshake port.

    The handshake port is an FNV-1a hash of the ipc:// path folded into a
    10000-wide band, so distinct workers can collide. Two engines dialing the
    same tcp port would cross their handshakes, so reject it up front with a
    message naming the colliding URLs instead of failing opaquely at bind time.
    """
    seen: dict[int, str] = {}
    for port in ports:
        ipc_url = _zmq_ipc_url(port)
        handshake_port = _zmq_handshake_port(ipc_url)
        if handshake_port in seen:
            raise ValueError(
                f"ZMQ handshake port collision on {handshake_port}: "
                f"{seen[handshake_port]} and {ipc_url} derive the same port. "
                "Adjust --worker-base-port so the worker ipc paths differ."
            )
        seen[handshake_port] = ipc_url


# ---------------------------------------------------------------------------
# WorkerLauncher ABC + backend implementations
# ---------------------------------------------------------------------------


class WorkerLauncher(ABC):
    """Abstract base class for backend worker launchers."""

    @abstractmethod
    def build_command(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int
    ) -> list[str]:
        """Build the CLI command list to launch a worker."""
        ...

    def health_check(self, args: argparse.Namespace, host: str, port: int, timeout: float) -> bool:
        """Return True when the worker at host:port is healthy."""
        mode = getattr(args, "connection_mode", "grpc")
        if mode == "grpc":
            return _grpc_health_check(host, port, timeout)
        if mode == "zmq":
            # SMG binds the ZMQ sockets and the engine dials in; there is no
            # host:port to probe. The router's own health checker gates
            # readiness, so the launcher just proceeds to start the router.
            return True
        return _http_health_check(f"http://{host}:{port}/health", timeout)

    def worker_url(self, args: argparse.Namespace, host: str, port: int) -> str:
        """Return the URL used by the router to reach this worker."""
        mode = getattr(args, "connection_mode", "grpc")
        if mode == "grpc":
            return f"grpc://{host}:{port}"
        if mode == "zmq":
            return _zmq_ipc_url(port)
        return f"http://{host}:{port}"

    def _get_tp_size(self, args: argparse.Namespace) -> int:
        """Return tensor-parallel size for GPU assignment. Default 1."""
        return 1

    def gpu_env(self, args: argparse.Namespace, dp_rank: int, env: dict | None = None) -> dict:
        """Build env dict with CUDA_VISIBLE_DEVICES for this worker's dp_rank.

        If CUDA_VISIBLE_DEVICES is already set in the environment, it is treated
        as the available GPU pool and indexed into by dp_rank/tp_size.  This
        allows users to control GPU assignment externally while still supporting
        multi-worker partitioning.
        """
        env = dict(env) if env is not None else os.environ.copy()
        tp_size = self._get_tp_size(args)
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
        base_idx = dp_rank * tp_size

        visible = env.get("CUDA_VISIBLE_DEVICES", "")
        available_gpus = [g.strip() for g in visible.split(",") if g.strip()]

        if available_gpus:
            if base_idx + tp_size > len(available_gpus):
                raise ValueError(
                    f"CUDA_VISIBLE_DEVICES has {len(available_gpus)} GPU(s) but "
                    f"dp_rank={dp_rank} with tp_size={tp_size} requires "
                    f"index {base_idx}..{base_idx + tp_size - 1}"
                )
            gpu_ids = available_gpus[base_idx : base_idx + tp_size]
        else:
            gpu_ids = [str(g) for g in range(base_idx, base_idx + tp_size)]

        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _filter_backend_args(self, backend_args: list[str], filter_args: list[str]) -> list[str]:
        """Filter out args from backend_args that are already set by the launcher.

        Handles ``--key=value``, ``--key value``, and boolean flags. A filtered
        key in ``--key value`` form only consumes the following token when it
        actually looks like a value (not another ``-``-prefixed flag), so a
        filtered boolean flag does not swallow the argument after it.
        """
        filtered = []
        skip_next = False
        for i, arg in enumerate(backend_args):
            if skip_next:
                skip_next = False
                continue
            key = arg.split("=", 1)[0]
            if key in filter_args:
                if "=" not in arg:
                    nxt = backend_args[i + 1] if i + 1 < len(backend_args) else None
                    if nxt is not None and not nxt.startswith("-"):
                        skip_next = True  # value is the next token
                continue
            filtered.append(arg)
        return filtered

    def _backend_arg_defaults(
        self, backend_args: list[str], defaults: dict[str, list[str]]
    ) -> list[str]:
        """Expand each default the caller did not set themselves.

        Unlike ``_filter_backend_args``, which strips flags the launcher owns
        outright, these are engine defaults SMG needs for the wire to behave as
        the API promises but an operator may legitimately override.
        """
        supplied = {arg.split("=", 1)[0] for arg in backend_args}
        return [
            token for flag, tokens in defaults.items() if flag not in supplied for token in tokens
        ]

    def launch(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int, env: dict
    ) -> subprocess.Popen:
        """Launch the worker subprocess."""
        cmd = self.build_command(args, backend_args, host, port)
        logger.info("Launching worker with command: %s", " ".join(cmd))

        return subprocess.Popen(
            cmd,
            start_new_session=True,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )


class SglangWorkerLauncher(WorkerLauncher):
    """Launcher for sglang inference workers."""

    def _get_tp_size(self, args: argparse.Namespace) -> int:
        return getattr(args, "tensor_parallel_size", 1)

    def build_command(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int
    ) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            getattr(args, "model_path", ""),
            "--host",
            host,
            "--port",
            str(port),
        ]
        if getattr(args, "connection_mode", "grpc") == "grpc":
            cmd.append("--grpc-mode")

        if getattr(args, "connection_mode", "grpc") == "http" and getattr(
            args, "enable_token_usage_details", False
        ):
            cmd.append("--enable-cache-report")

        cmd.extend(self._filter_backend_args(backend_args, ["--model-path", "--host", "--port"]))
        return cmd


class VllmWorkerLauncher(WorkerLauncher):
    """Launcher for vLLM inference workers."""

    def _get_tp_size(self, args: argparse.Namespace) -> int:
        return getattr(args, "tensor_parallel_size", 1)

    def build_command(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int
    ) -> list[str]:
        if getattr(args, "connection_mode", "grpc") == "zmq":
            return self._build_zmq_command(args, backend_args, port)

        vllm_entry_points = (
            "vllm.entrypoints.grpc_server"
            if args.connection_mode == "grpc"
            else "vllm.entrypoints.openai.api_server"
        )

        cmd = [
            sys.executable,
            "-m",
            vllm_entry_points,
            "--model",
            getattr(args, "model", ""),
            "--host",
            host,
            "--port",
            str(port),
        ]
        if getattr(args, "connection_mode", "grpc") == "http" and getattr(
            args, "enable_token_usage_details", False
        ):
            cmd.append("--enable-prompt-tokens-details")

        cmd.extend(
            self._filter_backend_args(backend_args, ["--model", "--host", "--port", "--uds"])
        )

        return cmd

    def _build_zmq_command(
        self, args: argparse.Namespace, backend_args: list[str], port: int
    ) -> list[str]:
        """Launch a headless vLLM EngineCore group that dials SMG's ZMQ handshake.

        SMG (the router) binds the tcp handshake + ipc data-plane sockets it
        derives from the ipc:// worker URL; the engines connect in. An
        engine-level ``--data-parallel-size N`` (after ``--``) launches a
        grouped worker: N engines on one socket set, balanced by the
        connector. The default stays one standalone engine per worker;
        running several workers is dense data parallelism.
        """
        engine_dp = _backend_arg_int(backend_args, "--data-parallel-size", 1)
        rpc_port = _zmq_handshake_port(_zmq_ipc_url(port))
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            getattr(args, "model", ""),
            "--headless",
            "--data-parallel-size",
            str(engine_dp),
            "--data-parallel-size-local",
            str(engine_dp),
            "--data-parallel-address",
            "127.0.0.1",
            "--data-parallel-rpc-port",
            str(rpc_port),
        ]
        cmd.extend(
            self._filter_backend_args(
                backend_args,
                [
                    "--model",
                    "--headless",
                    "--data-parallel-size",
                    "--data-parallel-size-local",
                    "--data-parallel-address",
                    "--data-parallel-rpc-port",
                ],
            )
        )
        return cmd


class TokenspeedWorkerLauncher(WorkerLauncher):
    """Launcher for TokenSpeed inference workers (ZMQ direct-backend only)."""

    def _get_tp_size(self, args: argparse.Namespace) -> int:
        return getattr(args, "tensor_parallel_size", 1) or 1

    def build_command(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int
    ) -> list[str]:
        if getattr(args, "connection_mode", "grpc") != "zmq":
            raise ValueError(
                "TokenSpeed backend only supports --connection-mode zmq "
                "(the headless engine speaks the ZMQ direct-backend wire)"
            )
        return self._build_zmq_command(args, backend_args, port)

    def _build_zmq_command(
        self, args: argparse.Namespace, backend_args: list[str], port: int
    ) -> list[str]:
        """Launch a headless TokenSpeed scheduler group that dials SMG's ZMQ handshake.

        SMG (the router) binds the tcp handshake + ipc data-plane sockets it
        derives from the ipc:// worker URL; the engines connect in. An
        engine-level ``--data-parallel-size N`` (after ``--``) launches a
        grouped worker: N ranks on one socket set, each dialing with its own
        identity (``--zmq-engine-index`` is the group's base). The default
        stays one standalone engine per worker.

        Two ports are the launcher's to own, or co-located workers collide:

        - ``--port`` seeds TokenSpeed's whole derived control-plane port
          cluster (torch.distributed store at ``port + 233``, and neighbors).
          Left at the engine default, every worker on the host derives the
          same cluster, and back-to-back engine restarts race the previous
          process's teardown (EADDRINUSE on the distributed store).
        - ``--dist-init-addr`` pins that store explicitly. TokenSpeed derives
          it from ``--port`` at dp==1 but refuses to guess for dp>1; passing
          the same derivation it would use keeps one port layout for both.
        """
        rpc_port = _zmq_handshake_port(_zmq_ipc_url(port))
        # Mirrors TokenSpeed's own dp==1 derivation (ZMQ_TCP_PORT_DELTA);
        # reflected below the u16 ceiling instead of wrapping into low ports.
        dist_port = port + 233 if port + 233 <= 65535 else port - 233
        # Hop over the SMG handshake band: a dist port inside it can land on
        # a worker's rpc listener (this worker's included — the band is a hash
        # of the ipc path). The +233 branch enters the band only from below,
        # so one span-wide hop exits it for good; the -233 branch starts far
        # above the band.
        if (
            _ZMQ_HANDSHAKE_PORT_BASE
            <= dist_port
            < _ZMQ_HANDSHAKE_PORT_BASE + _ZMQ_HANDSHAKE_PORT_SPAN
        ):
            dist_port += _ZMQ_HANDSHAKE_PORT_SPAN
        cmd = [
            sys.executable,
            "-m",
            "tokenspeed.cli",
            "serve",
            "--headless",
            "--model",
            getattr(args, "model", ""),
            "--port",
            str(port),
            "--dist-init-addr",
            f"127.0.0.1:{dist_port}",
            "--data-parallel-address",
            "127.0.0.1",
            "--data-parallel-rpc-port",
            str(rpc_port),
            "--zmq-engine-index",
            "0",
        ]
        cmd.extend(
            self._backend_arg_defaults(
                backend_args,
                {
                    # ``tool_choice=required``/``{function}`` and
                    # ``response_format=json_schema`` reach the engine as
                    # constraints on the wire, but TokenSpeed drops them without
                    # complaint unless a grammar backend is configured, and its
                    # default is none. Without this an unenforced constraint is
                    # silently answered as freeform text.
                    "--grammar-backend": ["--grammar-backend", "xgrammar"],
                    # Output logprobs default off, so ``logprobs=true`` would
                    # come back null rather than with per-token data.
                    "--enable-output-logprobs": ["--enable-output-logprobs"],
                },
            )
        )
        cmd.extend(
            self._filter_backend_args(
                backend_args,
                [
                    "--model",
                    "--headless",
                    "--port",
                    "--dist-init-addr",
                    "--data-parallel-address",
                    "--data-parallel-rpc-port",
                    "--zmq-engine-index",
                ],
            )
        )
        return cmd


class TrtllmWorkerLauncher(WorkerLauncher):
    """Launcher for TensorRT-LLM inference workers (gRPC mode only).

    Uses ``python3 -m tensorrt_llm.commands.serve <model> --grpc ...``.
    See https://github.com/NVIDIA/TensorRT-LLM/pull/11037
    """

    def _get_tp_size(self, args: argparse.Namespace) -> int:
        """Get tensor parallel size from args or config file.

        Priority: args.tp_size > args.tensor_parallel_size > config file > default(1)

        Raises:
            FileNotFoundError: If --config path does not exist.
            yaml.YAMLError: If --config file contains invalid YAML.
        """
        # Try --tp-size argument first
        tp_size = getattr(args, "tp_size", None)
        if tp_size is not None:
            return tp_size

        # Try --tensor-parallel-size (vLLM-style naming)
        tp_size = getattr(args, "tensor_parallel_size", None)
        if tp_size is not None:
            return tp_size

        # Try reading from config YAML file — let I/O and parse errors propagate
        # so misconfigurations are caught early instead of silently using tp=1.
        config_path = getattr(args, "config", None)
        if config_path:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
            if config and "tensor_parallel_size" in config:
                return int(config["tensor_parallel_size"])
            if config and "tp_size" in config:
                return int(config["tp_size"])
            logger.warning(
                "Config %s does not contain tensor_parallel_size or tp_size, defaulting to 1",
                config_path,
            )

        return 1

    def build_command(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int
    ) -> list[str]:
        if getattr(args, "connection_mode", "grpc") != "grpc":
            raise ValueError("TensorRT-LLM backend only supports grpc connection mode")

        cmd = [
            sys.executable,
            "-m",
            "tensorrt_llm.commands.serve",
            getattr(args, "model_path", ""),
            "--grpc",
            "--host",
            host,
            "--port",
            str(port),
        ]

        # Add optional config file
        # TRT-LLM Click options use underscores (e.g. --tensor_parallel_size)
        # while SGLang/vLLM use hyphens. Normalize so users can pass either form.
        normalized = [
            "--" + a[2:].replace("-", "_") if a.startswith("--") else a for a in backend_args
        ]
        cmd.extend(
            self._filter_backend_args(normalized, ["--model", "--model_path", "--host", "--port"])
        )

        return cmd


BACKEND_LAUNCHERS: dict[str, type[WorkerLauncher]] = {
    "sglang": SglangWorkerLauncher,
    "vllm": VllmWorkerLauncher,
    "trtllm": TrtllmWorkerLauncher,
    "tokenspeed": TokenspeedWorkerLauncher,
}


# ---------------------------------------------------------------------------
# Health check utilities
# ---------------------------------------------------------------------------


def _http_health_check(url: str, timeout: float) -> bool:
    """GET the URL and return True on HTTP 200."""
    try:
        import urllib.request

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception as e:
        logger.debug("HTTP health check for %s failed: %s", url, e)
        return False


def _grpc_health_check(host: str, port: int, timeout: float) -> bool:
    """Standard gRPC health check with fallback to channel_ready for vLLM."""
    try:
        import grpc
        from grpc_health.v1 import health_pb2, health_pb2_grpc
    except ImportError:
        logger.debug("gRPC libraries not available for health check")
        return False

    try:
        channel = grpc.insecure_channel(f"{host}:{port}")
        try:
            stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest(service="")
            response = stub.Check(request, timeout=timeout)
            return response.status == health_pb2.HealthCheckResponse.SERVING
        finally:
            channel.close()
    except grpc.RpcError as e:
        # vLLM doesn't implement gRPC health service — fall back to channel ready
        if hasattr(e, "code") and e.code() == grpc.StatusCode.UNIMPLEMENTED:
            try:
                channel = grpc.insecure_channel(f"{host}:{port}")
                try:
                    grpc.channel_ready_future(channel).result(timeout=timeout)
                    return True
                finally:
                    channel.close()
            except Exception as fallback_err:
                logger.debug(
                    "gRPC channel_ready fallback for %s:%d failed: %s",
                    host,
                    port,
                    fallback_err,
                )
                return False
        logger.debug("gRPC health check for %s:%d failed: %s", host, port, e)
        return False
    except Exception as e:
        logger.debug("gRPC health check error for %s:%d: %s", host, port, e)
        return False


# ---------------------------------------------------------------------------
# Port discovery
# ---------------------------------------------------------------------------


def _find_available_ports(base_port: int, count: int) -> list[int]:
    """Find *count* available ports starting near *base_port*.

    Uses socket bind test (no sglang dependency).  Ports are spaced with a
    small random offset to reduce collisions across concurrent launches.
    """
    ports: list[int] = []
    candidate = base_port
    while len(ports) < count:
        if _is_port_available(candidate):
            ports.append(candidate)
            candidate += random.randint(1, 5)
        else:
            candidate += 1
        if candidate > 65535:
            raise RuntimeError(f"Could not find {count} available ports starting from {base_port}")
    return ports


def _is_port_available(port: int) -> bool:
    """Return True if *port* is free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Argument adders (backend-specific CLI arguments)
# ---------------------------------------------------------------------------


def _add_sglang_args(parser: argparse.ArgumentParser) -> None:
    """Add sglang-specific arguments."""
    try:
        from sglang.srt.server_args import ServerArgs

        ServerArgs.add_cli_args(parser)
    except ImportError:
        parser.error("sglang is not installed. Install it with: pip install sglang")


def _add_vllm_args(parser: argparse.ArgumentParser) -> None:
    """Add vLLM engine CLI arguments."""
    try:
        from vllm.engine import arg_utils as vllm_arg_utils

        engine_args_cls = getattr(vllm_arg_utils, "AsyncEngineArgs", None) or getattr(
            vllm_arg_utils, "EngineArgs", None
        )
        if engine_args_cls is None:
            raise ImportError("vllm.engine.arg_utils is missing EngineArgs")

        engine_args_cls.add_cli_args(parser)
    except ImportError:
        parser.error("vllm is not installed. Install it with: pip install vllm")


def _add_vllm_frontend_args(parser: argparse.ArgumentParser) -> None:
    """Add vLLM OpenAI frontend arguments."""
    try:
        from vllm.entrypoints.openai import cli_args as vllm_openai_cli_args
    except ImportError:
        vllm_openai_cli_args = None

    frontend_args_cls = (
        getattr(vllm_openai_cli_args, "FrontendArgs", None)
        if vllm_openai_cli_args is not None
        else None
    )
    if frontend_args_cls is not None:
        frontend_args_cls.add_cli_args(parser)


def _add_trtllm_stub_args(parser: argparse.ArgumentParser) -> None:
    """Add TensorRT-LLM specific arguments.

    Note: TensorRT-LLM doesn't provide a centralized argument manager like
    vLLM's EngineArgs. We manually add the most commonly used arguments.
    TP size is read from the config file, not passed as CLI argument.
    """
    group = parser.add_argument_group("TensorRT-LLM Options")
    group.add_argument(
        "--model",
        "--model-path",
        dest="model_path",
        type=str,
        help="Model path (HuggingFace ID or local path)",
    )
    group.add_argument("--tp_size", type=int, help="Tensor parallel size (overrides config file)")


def _add_tokenspeed_stub_args(parser: argparse.ArgumentParser) -> None:
    """Add TokenSpeed-specific arguments.

    TokenSpeed args are passed through verbatim to the engine command; only the
    flags the launcher itself consumes are declared here (parse_serve_args uses
    parse_known_args for this backend, like trtllm).
    """
    group = parser.add_argument_group("TokenSpeed Options")
    group.add_argument(
        "--model",
        type=str,
        help="Model path (HuggingFace ID or local path)",
    )
    group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (for per-worker GPU assignment)",
    )


BACKEND_ARG_ADDERS = {
    "sglang": _add_sglang_args,
    "vllm": _add_vllm_args,
    "trtllm": _add_trtllm_stub_args,
    "tokenspeed": _add_tokenspeed_stub_args,
}

BACKEND_CHOICES = list(BACKEND_ARG_ADDERS.keys())
DEFAULT_BACKEND = os.getenv("SMG_DEFAULT_BACKEND", "sglang")


# ---------------------------------------------------------------------------
# Serve argument parsing (two-pass)
# ---------------------------------------------------------------------------


def add_serve_args(parser: argparse.ArgumentParser) -> None:
    """Add serve-specific arguments (not from any backend)."""
    group = parser.add_argument_group("Serve Options")
    group.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=BACKEND_CHOICES,
        help=f"Inference backend to use (default: {DEFAULT_BACKEND})",
    )
    group.add_argument(
        "--connection-mode",
        default="grpc",
        choices=["grpc", "http", "zmq"],
        help=(
            "Connection mode for workers (default: grpc). Note: trtllm only "
            "supports grpc, tokenspeed only supports zmq, and zmq is otherwise "
            "only supported for the vllm backend"
        ),
    )
    # Router host/port - may be overridden by backend (e.g. sglang)
    group.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the router (default: 127.0.0.1)",
    )
    group.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the router (default: 8080)",
    )
    # Data parallel size - may be overridden by backend
    group.add_argument(
        "--data-parallel-size",
        "--dp-size",
        type=int,
        default=1,
        dest="data_parallel_size",
        help="Data parallel size (number of worker replicas)",
    )
    group.add_argument(
        "--worker-host",
        default="127.0.0.1",
        help="Host for worker processes (default: 127.0.0.1)",
    )
    group.add_argument(
        "--worker-base-port",
        type=int,
        default=31000,
        help="Base port for workers (default: 31000)",
    )
    group.add_argument(
        "--worker-startup-timeout",
        type=int,
        default=300,
        help="Seconds to wait for workers to become healthy (default: 300)",
    )
    group.add_argument(
        "--enable-token-usage-details",
        action="store_true",
        help="Enable detailed token usage reporting (if supported by backend and router)",
    )


def _import_backend_args(backend: str, parser: argparse.ArgumentParser) -> None:
    """Conditionally import and add backend-native args to parser."""
    BACKEND_ARG_ADDERS[backend](parser)


def parse_serve_args(
    argv: list[str] | None = None,
) -> tuple[str, argparse.Namespace, list[str]]:
    """Two-pass argument parsing for serve command.

    Pass 1: Parse with only serve + router args (parse_known_args). We get
    --backend and known flags; any remaining argv tokens are left as
    backend_args (no backend-specific parser loaded yet).
    Pass 2: Build full parser with serve + backend-specific + router args,
    then parse_args(argv) to get the full namespace. backend_args from pass 1
    are returned unchanged for the launcher to pass through to worker commands.

    Returns:
        Tuple of (backend_name, parsed_namespace, backend_args).
        backend_args: argv tokens not recognized in pass 1 (e.g. --config path).
    """
    if argv is None:
        argv = []

    # Pass 1: serve + router args only; unknown tokens become backend_args
    pre_parser = argparse.ArgumentParser(add_help=False)
    add_serve_args(pre_parser)
    RouterArgs.add_cli_args(pre_parser, use_router_prefix=True, exclude_host_port=True)
    serve_router_args, backend_args = pre_parser.parse_known_args(argv)
    backend = serve_router_args.backend

    # ZMQ direct-backend is a same-host engine connection; only vLLM EngineCore
    # and TokenSpeed speak a supported ZMQ wire protocol.
    if serve_router_args.connection_mode == "zmq" and backend not in ("vllm", "tokenspeed"):
        pre_parser.error(
            "connection-mode zmq is only supported for the vllm and tokenspeed "
            f"backends, not {backend}"
        )

    # Pass 2: full parser with backend-specific args; resolve so backend can override
    parser = argparse.ArgumentParser(
        description=f"Launch {backend} worker(s) + gateway router",
        conflict_handler="resolve",
    )
    add_serve_args(parser)
    _import_backend_args(backend, parser)
    if backend == "vllm" and serve_router_args.connection_mode == "http":
        logger.debug("Adding vLLM OpenAI frontend CLI arguments for HTTP workers.")
        _add_vllm_frontend_args(parser)
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)

    # trtllm/tokenspeed only declare stub args (no full engine parser); unknown
    # tokens stay in backend_args and are passed through to the worker command.
    if backend in ("trtllm", "tokenspeed"):
        args, _ = parser.parse_known_args(argv)
    else:
        args = parser.parse_args(argv)

    return backend, args, backend_args


# ---------------------------------------------------------------------------
# ServeOrchestrator
# ---------------------------------------------------------------------------

_WORKER_SHUTDOWN_TIMEOUT = 30


class ServeOrchestrator:
    """Coordinate worker launch, health checking, router startup, and shutdown."""

    def __init__(self, backend: str, args: argparse.Namespace, backend_args: list[str]):
        self.backend = backend
        self.args = args
        self.backend_args = backend_args
        self.launcher: WorkerLauncher = BACKEND_LAUNCHERS[backend]()
        self.workers: list[tuple[subprocess.Popen, int]] = []
        self._shutting_down = False

    # -- public API ---------------------------------------------------------

    def run(self) -> None:
        """Full lifecycle: launch workers → health check → start router."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup_workers)
        try:
            self._launch_workers()
            self._wait_healthy()
            router_args = self._build_router_args()
            launch_router(router_args)
        finally:
            self._cleanup_workers()

    # -- internal -----------------------------------------------------------

    def _launch_workers(self) -> None:
        ports = _find_available_ports(self.args.worker_base_port, self.args.data_parallel_size)
        if getattr(self.args, "connection_mode", "grpc") == "zmq":
            _reject_handshake_port_collisions(ports)
        host = self.args.worker_host
        for dp_rank, port in enumerate(ports):
            env = self.launcher.gpu_env(self.args, dp_rank)
            proc = self.launcher.launch(self.args, self.backend_args, host, port, env)
            self.workers.append((proc, port))
            logger.info(
                "Launched %s worker on %s:%d (pid %d, GPUs: %s)",
                self.backend,
                host,
                port,
                proc.pid,
                env["CUDA_VISIBLE_DEVICES"],
            )

    def _wait_healthy(self) -> None:
        host = self.args.worker_host
        for proc, port in self.workers:
            deadline = time.monotonic() + self.args.worker_startup_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Worker on port {port} exited with code {proc.returncode}")
                if self.launcher.health_check(self.args, host, port, timeout=5.0):
                    logger.info("Worker on %s:%d is healthy", host, port)
                    break
                time.sleep(2)
            else:
                raise TimeoutError(
                    f"Worker on port {port} not healthy within {self.args.worker_startup_timeout}s"
                )

    def _build_router_args(self) -> RouterArgs:
        worker_urls = [
            self.launcher.worker_url(self.args, self.args.worker_host, port)
            for _, port in self.workers
        ]
        router_args = RouterArgs.from_cli_args(self.args, use_router_prefix=True)
        router_args.worker_urls = worker_urls
        # The ZMQ handshake is shared across engine runtimes, so the router
        # cannot probe the wire protocol; forward the serve backend so the Rust
        # side stamps the startup workers' runtime. (RouterArgs.backend
        # otherwise keeps its own --router-backend value, which serve's
        # --backend does not reach.)
        if getattr(self.args, "connection_mode", "grpc") == "zmq":
            router_args.backend = self.backend
            # A grouped vLLM launch (engine-level --data-parallel-size N after
            # `--`) needs the router to await N engines per worker handshake.
            engine_dp = _backend_arg_int(self.backend_args, "--data-parallel-size", 1)
            if engine_dp > 1 and self.backend == "vllm":
                router_args.zmq_engine_count = engine_dp
        # Router-level retries and circuit breaker are redundant when there is a
        # single worker — per-worker resilience already handles failures — so
        # disable them by default for dp<=1. Users who want them must run dp>1.
        if self.args.data_parallel_size <= 1:
            if not router_args.disable_retries:
                logger.info(
                    "dp=%d: disabling router-level retries (per-worker resilience handles failures)",
                    self.args.data_parallel_size,
                )
                router_args.disable_retries = True
            if not router_args.disable_circuit_breaker:
                logger.info(
                    "dp=%d: disabling router-level circuit breaker (per-worker resilience handles failures)",
                    self.args.data_parallel_size,
                )
                router_args.disable_circuit_breaker = True
            # A single backend has nothing to balance across, so force the
            # passthrough policy (overriding any --policy). It skips load
            # monitoring and KV-event subscription, dropping that overhead.
            if router_args.policy != "passthrough":
                logger.info(
                    "dp=%d: forcing router policy to passthrough (single backend; "
                    "skips load monitoring and KV-event subscription)",
                    self.args.data_parallel_size,
                )
                router_args.policy = "passthrough"
        return router_args

    def _signal_handler(self, signum: int, frame: object) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Received signal %d, shutting down workers…", signum)
        self._cleanup_workers()
        sys.exit(128 + signum)

    def _cleanup_workers(self) -> None:
        """SIGTERM all worker process groups, wait, then SIGKILL stragglers."""
        if not self.workers:
            return

        # Send SIGTERM to each process group
        for proc, port in self.workers:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        # Wait up to _WORKER_SHUTDOWN_TIMEOUT seconds for graceful exit
        deadline = time.monotonic() + _WORKER_SHUTDOWN_TIMEOUT
        for proc, port in self.workers:
            remaining = max(0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve_main(argv: list[str] | None = None) -> None:
    """Parse serve args, create orchestrator, and run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    backend, args, backend_args = parse_serve_args(argv)
    orchestrator = ServeOrchestrator(backend, args, backend_args)
    orchestrator.run()
