"""Shared lifecycle hook for the Rust WorkerControl server.

The extension import is deliberately lazy: engine servicers that do not set
``SMG_WORKER_CONTROL_BIND_ADDRESS`` keep their existing dependency surface.
Explicitly enabling the control plane is fail-closed.
"""

from __future__ import annotations

import importlib.metadata
import os
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _optional_env(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return value.strip() if value and value.strip() else None


def _bool_env(environ: Mapping[str, str], name: str) -> bool:
    value = _optional_env(environ, name)
    if value is None:
        return False
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


@dataclass
class WorkerControlLifecycle:
    """Small engine-independent wrapper around the PyO3 server."""

    server: Any

    @classmethod
    def start_from_env(
        cls,
        *,
        engine_type: str,
        model_ids: Sequence[str],
        features: Sequence[str],
        max_concurrent_requests: int = 0,
        engine_attributes: Mapping[str, str] | None = None,
        engine_distribution: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> WorkerControlLifecycle | None:
        environ = os.environ if environ is None else environ
        bind_address = _optional_env(environ, "SMG_WORKER_CONTROL_BIND_ADDRESS")
        if bind_address is None:
            return None

        engine_endpoint = _optional_env(environ, "SMG_WORKER_ENGINE_ENDPOINT")
        if engine_endpoint is None:
            raise ValueError(
                "SMG_WORKER_ENGINE_ENDPOINT is required when the Worker control plane is enabled"
            )
        worker_id = _optional_env(environ, "SMG_WORKER_ID") or socket.gethostname()
        instance_id = _optional_env(environ, "SMG_WORKER_INSTANCE_ID")
        zone = _optional_env(environ, "SMG_WORKER_ZONE") or ""
        inference_enabled = _bool_env(environ, "SMG_WORKER_INFERENCE_ENABLED")
        engine_transport = _optional_env(environ, "SMG_WORKER_ENGINE_TRANSPORT") or "grpc"
        zmq_handshake_address = _optional_env(environ, "SMG_WORKER_ZMQ_HANDSHAKE_ADDRESS")
        engine_count_raw = _optional_env(environ, "SMG_WORKER_ENGINE_COUNT") or "1"
        try:
            engine_count = int(engine_count_raw)
        except ValueError as error:
            raise ValueError("SMG_WORKER_ENGINE_COUNT must be a positive integer") from error
        if engine_count <= 0:
            raise ValueError("SMG_WORKER_ENGINE_COUNT must be a positive integer")
        if max_concurrent_requests < 0:
            # Clamping to 0 would make an invalid limit indistinguishable from
            # "unlimited", the default -- fail at startup instead.
            raise ValueError("max_concurrent_requests must be non-negative")

        try:
            from smg.worker import WorkerControlServer
        except ImportError as error:
            raise RuntimeError(
                "the SMG Python extension is required when the Worker control plane is enabled"
            ) from error

        engine_version = ""
        if engine_distribution:
            try:
                engine_version = importlib.metadata.version(engine_distribution)
            except importlib.metadata.PackageNotFoundError:
                pass

        server = WorkerControlServer(
            bind_address=bind_address,
            worker_id=worker_id,
            instance_id=instance_id,
            hostname=socket.gethostname(),
            zone=zone,
            engine_type=engine_type,
            engine_version=engine_version,
            engine_endpoint=engine_endpoint,
            model_ids=list(model_ids),
            features=list(features),
            max_concurrent_requests=max_concurrent_requests,
            inference_enabled=inference_enabled,
            engine_attributes=dict(engine_attributes or {}),
            engine_transport=engine_transport,
            zmq_handshake_address=zmq_handshake_address,
            engine_count=engine_count,
        )
        return cls(server=server)

    @property
    def running(self) -> bool:
        return bool(self.server.running)

    @property
    def last_error(self) -> str | None:
        return self.server.last_error

    def mark_serving(self) -> None:
        self.server.set_health("serving", "ready")

    def mark_draining(self) -> None:
        self.server.set_health("draining", "draining")

    def mark_not_serving(self, message: str = "stopped") -> None:
        self.server.set_health("not_serving", message)

    def stop(self, timeout_secs: float = 5.0) -> None:
        self.server.stop(timeout_secs)
