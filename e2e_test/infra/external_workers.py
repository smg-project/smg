"""External (pre-started) worker pool configuration and attachment.

Allows ``WorkerPool`` to attach the gateway under test to already-running
HTTP or gRPC inference workers (for example, a long-lived engine pool on a
dev host or CI node) instead of cold-starting and tearing down an engine for
every test run.

Configuration is opt-in via ``E2E_EXTERNAL_WORKERS``, which accepts either an
inline JSON payload or a path to a JSON file (optionally prefixed with ``@``).
Two top-level JSON shapes are accepted:

1. A JSON object::

       {
           "workers": [
               {
                   "engine": "vllm",
                   "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                   "mode": "grpc",
                   "urls": ["grpc://10.0.0.5:30000"]
               },
               {
                   "engine": "vllm",
                   "model_id": "Qwen/Qwen2.5-7B-Instruct",
                   "url": "grpc://10.0.0.5:30001"
               }
           ],
           "flush_cache": true,
           "health_timeout": 5.0,
           "fallback_to_local": true
       }

2. A bare JSON list of worker entries (equivalent to ``{"workers": [...]}``
   with default top-level options).

Only ``REGULAR`` HTTP and gRPC workers are eligible for external attachment.
ZMQ workers, disaggregated (PD/EPD) workers, and tests requesting unmatched
topologies or ``extra_engine_args`` fall back cleanly to local launches.
External workers are health-checked before attachment and are never stopped
or signalled on eviction or session teardown.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .constants import (
    ENV_EXTERNAL_WORKERS,
    ENV_EXTERNAL_WORKERS_FLUSH_CACHE,
    MM_PROCESSING_WORKER,
    SERVICER_IMPLS,
    ConnectionMode,
    WorkerType,
    get_mm_processing,
    get_servicer_impl,
)
from .worker import Worker

logger = logging.getLogger(__name__)

DEFAULT_EXTERNAL_HEALTH_TIMEOUT = 5.0
_SUPPORTED_EXTERNAL_MODES = frozenset({ConnectionMode.HTTP, ConnectionMode.GRPC})
_MM_PROCESSING_VALUES = frozenset({MM_PROCESSING_WORKER, "router"})
_TRUE_VALUES = frozenset({"1", "true", "yes"})
_FALSE_VALUES = frozenset({"0", "false", "no"})


@dataclass(frozen=True)
class ExternalWorkerEntry:
    """One pool specification of pre-started external worker endpoints."""

    engine: str
    model_id: str
    mode: ConnectionMode
    urls: tuple[str, ...]
    worker_type: WorkerType = WorkerType.REGULAR
    gpus: int | None = None
    extra_engine_args: tuple[str, ...] | None = None
    servicer_impl: str | None = None
    mm_processing: str | None = None
    flush_cache: bool | None = None
    health_timeout: float | None = None


def parse_external_worker_url(
    url: str,
    mode: ConnectionMode | None = None,
) -> tuple[str, ConnectionMode, str, int]:
    """Validate and normalize an external worker endpoint URL.

    Supports ``grpc://host:port`` for gRPC workers and ``http://host[:port]`` /
    ``https://host[:port]`` for HTTP workers. When ``mode`` is omitted it is
    inferred from the URL scheme; when provided it must agree with the scheme.

    Returns:
        Tuple of ``(normalized_url, resolved_mode, host, port)``.

    Raises:
        ValueError: If the URL is malformed, uses an unsupported scheme, lacks
            a valid host/port, or disagrees with ``mode``.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"External worker URL must be a non-empty string, got {url!r}")
    normalized = url.strip().rstrip("/")
    parsed = urlparse(normalized)
    scheme = parsed.scheme.lower()
    if scheme == "grpc":
        inferred = ConnectionMode.GRPC
    elif scheme in ("http", "https"):
        inferred = ConnectionMode.HTTP
    else:
        raise ValueError(
            f"Unsupported external worker URL scheme in {url!r}; "
            "expected grpc://, http://, or https://"
        )

    if mode is not None:
        if mode not in _SUPPORTED_EXTERNAL_MODES:
            raise ValueError(
                f"External workers only support HTTP and gRPC modes, not {mode.value!r}"
            )
        if mode != inferred:
            raise ValueError(
                f"External worker URL {url!r} (scheme {scheme!r}) does not match "
                f"configured mode {mode.value!r}"
            )

    host = parsed.hostname
    if not host:
        raise ValueError(f"External worker URL {url!r} is missing a hostname")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"External worker URL {url!r} has an invalid port: {exc}") from None

    if port is None:
        if scheme == "http":
            port = 80
        elif scheme == "https":
            port = 443
        else:
            raise ValueError(f"External gRPC worker URL {url!r} must include an explicit port")

    if not (1 <= port <= 65535):
        raise ValueError(f"External worker URL {url!r} port {port} is out of range 1..65535")

    return normalized, inferred, host, port


def _parse_bool_env(var_name: str) -> bool | None:
    """Parse an optional boolean environment variable, or return ``None`` if unset/blank."""
    raw = os.environ.get(var_name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if not normalized:
        return None
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{var_name}={raw!r} is not a valid boolean; use one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}"
    )


def _validate_positive_timeout(value: Any, field_name: str) -> float:
    """Validate that a timeout field is a positive finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number, got {value!r}")
    return float(value)


def _parse_entry(raw: Any, index: int) -> ExternalWorkerEntry:
    """Validate and parse a single external worker entry dict."""
    if not isinstance(raw, dict):
        raise ValueError(f"External worker entry [{index}] must be a JSON object")

    engine_raw = raw.get("engine")
    if not isinstance(engine_raw, str) or not engine_raw.strip():
        raise ValueError(f"External worker entry [{index}] requires a non-empty 'engine' string")
    engine = engine_raw.strip().lower()

    model_raw = raw.get("model_id")
    if not isinstance(model_raw, str) or not model_raw.strip():
        raise ValueError(f"External worker entry [{index}] requires a non-empty 'model_id' string")
    model_id = model_raw.strip()

    worker_type_raw = raw.get("worker_type", WorkerType.REGULAR.value)
    if not isinstance(worker_type_raw, str):
        raise ValueError(
            f"External worker entry [{index}] 'worker_type' must be a string, got {worker_type_raw!r}"
        )
    try:
        worker_type = WorkerType(worker_type_raw.strip().lower())
    except ValueError:
        raise ValueError(
            f"External worker entry [{index}] has invalid worker_type {worker_type_raw!r}"
        ) from None
    if worker_type != WorkerType.REGULAR:
        raise ValueError(
            f"External worker entry [{index}] uses unsupported worker_type {worker_type.value!r}; "
            "only 'regular' workers are supported for external attachment"
        )

    mode: ConnectionMode | None = None
    if raw.get("mode") is not None:
        mode_raw = raw["mode"]
        if not isinstance(mode_raw, str):
            raise ValueError(
                f"External worker entry [{index}] 'mode' must be a string, got {mode_raw!r}"
            )
        try:
            mode = ConnectionMode(mode_raw.strip().lower())
        except ValueError:
            raise ValueError(
                f"External worker entry [{index}] has invalid mode {mode_raw!r}"
            ) from None
        if mode not in _SUPPORTED_EXTERNAL_MODES:
            raise ValueError(
                f"External worker entry [{index}] uses unsupported mode {mode.value!r}; "
                "only 'http' and 'grpc' are supported for external attachment"
            )

    has_url = "url" in raw and raw["url"] is not None
    has_urls = "urls" in raw and raw["urls"] is not None
    if has_url and has_urls:
        raise ValueError(f"External worker entry [{index}] must specify 'url' or 'urls', not both")
    if has_url:
        raw_urls = [raw["url"]]
    elif has_urls:
        if not isinstance(raw["urls"], list) or not raw["urls"]:
            raise ValueError(
                f"External worker entry [{index}] 'urls' must be a non-empty list of URLs"
            )
        raw_urls = raw["urls"]
    else:
        raise ValueError(f"External worker entry [{index}] requires 'url' or 'urls'")

    normalized_urls: list[str] = []
    resolved_mode = mode
    for url in raw_urls:
        norm_url, inferred_mode, _host, _port = parse_external_worker_url(url, resolved_mode)
        if resolved_mode is None:
            resolved_mode = inferred_mode
        normalized_urls.append(norm_url)
    assert resolved_mode is not None

    gpus = raw.get("gpus")
    if gpus is not None:
        if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 1:
            raise ValueError(
                f"External worker entry [{index}] 'gpus' must be a positive integer or null, "
                f"got {gpus!r}"
            )

    extra_args_raw = raw.get("extra_engine_args")
    extra_engine_args: tuple[str, ...] | None = None
    if extra_args_raw is not None:
        if not isinstance(extra_args_raw, list) or not all(
            isinstance(arg, str) for arg in extra_args_raw
        ):
            raise ValueError(
                f"External worker entry [{index}] 'extra_engine_args' must be a list of strings"
            )
        extra_engine_args = tuple(extra_args_raw) if extra_args_raw else None

    servicer_impl_raw = raw.get("servicer_impl")
    servicer_impl: str | None = None
    if servicer_impl_raw is not None:
        if (
            not isinstance(servicer_impl_raw, str)
            or servicer_impl_raw.strip().lower() not in SERVICER_IMPLS
        ):
            raise ValueError(
                f"External worker entry [{index}] 'servicer_impl' must be one of "
                f"{SERVICER_IMPLS}, got {servicer_impl_raw!r}"
            )
        servicer_impl = servicer_impl_raw.strip().lower()

    mm_raw = raw.get("mm_processing")
    mm_processing: str | None = None
    if mm_raw is not None:
        if not isinstance(mm_raw, str) or mm_raw.strip().lower() not in _MM_PROCESSING_VALUES:
            raise ValueError(
                f"External worker entry [{index}] 'mm_processing' must be one of "
                f"{sorted(_MM_PROCESSING_VALUES)}, got {mm_raw!r}"
            )
        mm_processing = mm_raw.strip().lower()

    flush_cache = raw.get("flush_cache")
    if flush_cache is not None and not isinstance(flush_cache, bool):
        raise ValueError(f"External worker entry [{index}] 'flush_cache' must be a boolean or null")

    health_timeout: float | None = None
    if raw.get("health_timeout") is not None:
        health_timeout = _validate_positive_timeout(
            raw["health_timeout"], f"External worker entry [{index}] 'health_timeout'"
        )

    return ExternalWorkerEntry(
        engine=engine,
        model_id=model_id,
        mode=resolved_mode,
        urls=tuple(normalized_urls),
        worker_type=worker_type,
        gpus=gpus,
        extra_engine_args=extra_engine_args,
        servicer_impl=servicer_impl,
        mm_processing=mm_processing,
        flush_cache=flush_cache,
        health_timeout=health_timeout,
    )


@dataclass(frozen=True)
class ExternalWorkerRegistry:
    """Registry of externally managed HTTP/gRPC workers available for attachment."""

    entries: tuple[ExternalWorkerEntry, ...] = ()
    flush_cache: bool = True
    health_timeout: float = DEFAULT_EXTERNAL_HEALTH_TIMEOUT
    fallback_to_local: bool = True

    def __bool__(self) -> bool:
        """Return True when at least one external worker entry is registered."""
        return bool(self.entries)

    def match(
        self,
        *,
        model_id: str,
        engine: str,
        mode: ConnectionMode,
        count: int,
        worker_type: WorkerType = WorkerType.REGULAR,
        gpus: int | None = None,
        extra_engine_args: list[str] | tuple[str, ...] | None = None,
    ) -> list[ExternalWorkerEntry]:
        """Return entries compatible with the requested pool key, exact-count matches first."""
        if (
            not self.entries
            or worker_type != WorkerType.REGULAR
            or mode not in _SUPPORTED_EXTERNAL_MODES
            or count < 1
        ):
            return []

        norm_engine = engine.strip().lower()
        norm_extra = tuple(extra_engine_args) if extra_engine_args else None

        exact: list[ExternalWorkerEntry] = []
        superset: list[ExternalWorkerEntry] = []
        for entry in self.entries:
            if (
                entry.engine != norm_engine
                or entry.model_id != model_id
                or entry.mode != mode
                or entry.worker_type != worker_type
                or entry.gpus != gpus
                or entry.extra_engine_args != norm_extra
                or len(entry.urls) < count
            ):
                continue
            if entry.servicer_impl is not None:
                current_impl = (
                    get_servicer_impl(norm_engine)
                    if norm_engine in ("vllm", "sglang", "tokenspeed", "trtllm")
                    else None
                )
                if current_impl != entry.servicer_impl:
                    continue
            if entry.mm_processing is not None:
                current_mm = get_mm_processing() or "router"
                if current_mm != entry.mm_processing:
                    continue
            if len(entry.urls) == count:
                exact.append(entry)
            else:
                superset.append(entry)
        return exact + superset

    def attach(
        self,
        *,
        model_id: str,
        engine: str,
        mode: ConnectionMode,
        count: int = 1,
        worker_type: WorkerType = WorkerType.REGULAR,
        gpus: int | None = None,
        extra_engine_args: list[str] | tuple[str, ...] | None = None,
    ) -> list[Worker] | None:
        """Verify health and attach ``count`` external workers matching the requested key.

        Returns a list of ``count`` healthy externally-managed ``Worker`` objects
        on match, or ``None`` when no matching entry exists (or when all matching
        entries are unhealthy and ``fallback_to_local`` is enabled).

        Raises:
            RuntimeError: If at least one entry matched the key, fewer than
                ``count`` endpoints passed live health checks, and
                ``fallback_to_local`` is ``False``.
        """
        candidates = self.match(
            model_id=model_id,
            engine=engine,
            mode=mode,
            count=count,
            worker_type=worker_type,
            gpus=gpus,
            extra_engine_args=extra_engine_args,
        )
        if not candidates:
            return None

        unhealthy_urls: list[str] = []
        for entry in candidates:
            probe_timeout = (
                entry.health_timeout if entry.health_timeout is not None else self.health_timeout
            )
            should_flush = entry.flush_cache if entry.flush_cache is not None else self.flush_cache
            healthy: list[Worker] = []
            for url in entry.urls:
                norm_url, resolved_mode, host, port = parse_external_worker_url(url, entry.mode)
                worker = Worker(
                    model_id=model_id,
                    engine=engine,
                    port=port,
                    gpu_ids=[],
                    mode=resolved_mode,
                    worker_type=worker_type,
                    extra_engine_args=list(extra_engine_args) if extra_engine_args else None,
                    host=host,
                    external=True,
                    external_url=norm_url,
                    flush_cache=should_flush,
                    health_timeout=probe_timeout,
                )
                if worker.is_healthy(timeout=probe_timeout):
                    healthy.append(worker)
                    if len(healthy) == count:
                        logger.info(
                            "WorkerPool: attached %d external %s %s worker(s) for %s: %s",
                            count,
                            engine,
                            mode.value,
                            model_id,
                            [w.base_url for w in healthy],
                        )
                        return healthy
                else:
                    unhealthy_urls.append(norm_url)

        detail = (
            f"External workers matched ({engine}, {model_id}, {mode.value}, count={count}) "
            f"but failed health checks: {unhealthy_urls}"
        )
        if self.fallback_to_local:
            logger.warning("%s; falling back to local worker launch", detail)
            return None
        raise RuntimeError(detail)


def load_external_workers(raw: str | None = None) -> ExternalWorkerRegistry:
    """Load and validate external worker pool configuration.

    Args:
        raw: Raw configuration string (inline JSON or path to a JSON file,
            optionally prefixed with ``@``). When ``None``, reads
            ``E2E_EXTERNAL_WORKERS`` from the environment.

    Returns:
        Parsed :class:`ExternalWorkerRegistry` (empty when unset or blank).

    Raises:
        ValueError: When the configuration file cannot be read, contains
            invalid JSON, or fails schema validation.
    """
    spec = os.environ.get(ENV_EXTERNAL_WORKERS, "") if raw is None else raw
    spec = spec.strip()
    if not spec:
        return ExternalWorkerRegistry()

    if spec[0] in ("{", "["):
        source = ENV_EXTERNAL_WORKERS
        payload_text = spec
    else:
        path = Path(spec[1:] if spec.startswith("@") else spec)
        source = str(path)
        try:
            payload_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Failed to read external worker config from {path}: {exc}") from exc

    try:
        data = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in external worker config ({source}): {exc}") from exc

    flush_cache = True
    health_timeout = DEFAULT_EXTERNAL_HEALTH_TIMEOUT
    fallback_to_local = True

    raw_entries: list[Any]
    if isinstance(data, list):
        raw_entries = data
    elif isinstance(data, dict):
        workers_value = data.get("workers")
        if not isinstance(workers_value, list):
            raise ValueError(
                f"External worker config ({source}) object must contain a 'workers' list"
            )
        raw_entries = workers_value
        if "flush_cache" in data:
            if not isinstance(data["flush_cache"], bool):
                raise ValueError(
                    f"External worker config ({source}) 'flush_cache' must be a boolean"
                )
            flush_cache = data["flush_cache"]
        if "health_timeout" in data:
            health_timeout = _validate_positive_timeout(
                data["health_timeout"], f"External worker config ({source}) 'health_timeout'"
            )
        if "fallback_to_local" in data:
            if not isinstance(data["fallback_to_local"], bool):
                raise ValueError(
                    f"External worker config ({source}) 'fallback_to_local' must be a boolean"
                )
            fallback_to_local = data["fallback_to_local"]
    else:
        raise ValueError(
            f"External worker config ({source}) must be a JSON object or list of worker entries"
        )

    flush_override = _parse_bool_env(ENV_EXTERNAL_WORKERS_FLUSH_CACHE)
    if flush_override is not None:
        flush_cache = flush_override

    entries = tuple(_parse_entry(item, idx) for idx, item in enumerate(raw_entries))
    return ExternalWorkerRegistry(
        entries=entries,
        flush_cache=flush_cache,
        health_timeout=health_timeout,
        fallback_to_local=fallback_to_local,
    )
