"""Session-scoped worker pool for E2E tests.

Caches workers by ``(engine, model_id, mode, worker_type, count)`` so
consecutive test classes that need the same backend don't pay the multi-
minute worker startup cost on every class boundary.

The pool holds at most one *active* key at a time. Switching keys evicts
(stops) the cached workers before starting the new set — required because
GPU resources are exclusive. Combined with the collection-ordering hook in
``fixtures/hooks.py`` (which clusters items by backend/model), this keeps
the worker alive across every test class that uses the same backend.

ZMQ reuse retains the gateway together with its workers: the engines dial
that gateway's handshake sockets and cannot move to a new gateway. Only
explicitly compatible functional classes opt into this paired lifetime.

PD-disaggregation paths (prefill+decode) don't cache: they hold multiple
workers concurrently and the caller manages teardown. But they still go
through ``acquire()`` so the pool can evict any cached regular worker
first — otherwise the PD launch would race a still-running cached worker
for the same GPUs.

Lifecycle is managed via ``pytest_sessionfinish`` in ``fixtures/hooks.py``;
a module-level ``atexit`` handler covers the case where pytest exits
before that hook runs (SIGINT / ``pytest.exit``).
"""

from __future__ import annotations

import atexit
import logging
import threading

from .constants import DEFAULT_STARTUP_TIMEOUT, ConnectionMode, WorkerType
from .gateway import Gateway
from .process_utils import wait_for_workers_ready
from .worker import Worker, start_workers, stop_workers

logger = logging.getLogger(__name__)


# Key is (engine, model_id, mode, worker_type, count, gpus, extra_engine_args).
# ``count`` is part of the key because a class asking for count=2 after a
# count=1 class on the same backend would otherwise reuse a 1-worker entry and
# run with the wrong topology; ``gpus``/``extra_engine_args`` likewise change
# the launched topology (e.g. --data-parallel-size).
_PoolKey = tuple[str, str, ConnectionMode, WorkerType, int, int | None, tuple[str, ...] | None]


class WorkerPool:
    """One-slot worker cache shared across pytest classes.

    Not safe for concurrent use across pytest-xdist workers — each xdist
    worker would need its own pool with non-overlapping GPU offsets.
    Current CI runs sequentially on GPU runners, so a single-slot pool is
    sufficient.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: _PoolKey | None = None
        self._workers: list[Worker] = []
        self._gateway: Gateway | None = None
        self._gateway_key: tuple | None = None
        self._closed = False

    def acquire(
        self,
        *,
        model_id: str,
        engine: str,
        mode: ConnectionMode = ConnectionMode.HTTP,
        count: int = 1,
        worker_type: WorkerType = WorkerType.REGULAR,
        timeout: int = DEFAULT_STARTUP_TIMEOUT,
        log_dir: str | None = None,
        gpu_offset: int = 0,
        gpus: int | None = None,
        extra_engine_args: list[str] | None = None,
        wait_ready: bool = True,
        tp: int | None = None,
        kv_backend: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> list[Worker]:
        """Return ``count`` healthy workers for the given key.

        Reuses the cached set when the key matches AND every cached worker
        is still alive. Anything else (key mismatch, dead worker,
        non-REGULAR worker_type) evicts and starts fresh.

        Raises whatever ``start_workers`` raises on launch failure; the
        cache is left empty in that case.

        The lock is intentionally held across the blocking ``start_workers``
        call. CI runs sequentially today, so contention is a non-issue; if
        pytest-xdist is ever introduced each worker should get its own pool
        rather than competing for this one.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("WorkerPool has been closed")

            # Non-REGULAR workers (PD prefill/decode) aren't cached, but we
            # still have to release any cached regular worker first — it
            # holds the GPUs the caller is about to claim. ZMQ workers are
            # likewise uncached: the engine dials one gateway's handshake
            # sockets and cannot be reused by the next class's gateway.
            if worker_type != WorkerType.REGULAR or mode == ConnectionMode.ZMQ:
                if self._key is not None:
                    logger.info(
                        "WorkerPool: evicting %s to free GPUs for uncached %s %s/%s",
                        self._key,
                        mode.value,
                        worker_type,
                        model_id,
                    )
                    self._evict_locked()
                return start_workers(
                    model_id=model_id,
                    engine=engine,
                    mode=mode,
                    count=count,
                    worker_type=worker_type,
                    timeout=timeout,
                    log_dir=log_dir,
                    gpu_offset=gpu_offset,
                    gpus=gpus,
                    extra_engine_args=extra_engine_args,
                    wait_ready=wait_ready,
                    tp=tp,
                    kv_backend=kv_backend,
                    extra_env=extra_env,
                )

            if tp is not None or kv_backend is not None or extra_env is not None:
                raise ValueError(
                    "a per-leg tp, kv_backend or extra_env is only meaningful for "
                    "PD prefill/decode workers"
                )

            # REGULAR workers always start at gpu 0; ``gpu_offset`` is only
            # meaningful for non-REGULAR (PD decode) callers.
            key: _PoolKey = (
                engine,
                model_id,
                mode,
                worker_type,
                count,
                gpus,
                tuple(extra_engine_args) if extra_engine_args else None,
            )

            if self._key == key and all(w.is_alive() for w in self._workers):
                logger.info(
                    "WorkerPool: reusing %d cached worker(s) for %s",
                    count,
                    key,
                )
                return list(self._workers)

            if self._key is not None:
                reason = "key mismatch" if self._key != key else "dead worker"
                logger.info(
                    "WorkerPool: evicting %s to start %s (%s)",
                    self._key,
                    key,
                    reason,
                )
                self._evict_locked()

            new_workers = start_workers(
                model_id=model_id,
                engine=engine,
                mode=mode,
                count=count,
                worker_type=worker_type,
                timeout=timeout,
                log_dir=log_dir,
                gpus=gpus,
                extra_engine_args=extra_engine_args,
                wait_ready=wait_ready,
            )
            self._key = key
            self._workers = new_workers
            return list(new_workers)

    def acquire_zmq(
        self,
        *,
        model_id: str,
        model_path: str,
        engine: str,
        count: int,
        gpus: int | None,
        extra_engine_args: list[str] | None,
        gateway_config: dict,
        log_dir: str | None = None,
    ) -> Gateway:
        """Acquire one healthy ZMQ gateway/engine pair with identical config.

        Lane environment and model specifications are fixed for a pytest
        session. Class-varying engine and gateway options are matched exactly;
        no default or CLI argument normalization expands compatibility.
        """
        key: _PoolKey = (
            engine,
            model_id,
            ConnectionMode.ZMQ,
            WorkerType.REGULAR,
            count,
            gpus,
            tuple(extra_engine_args) if extra_engine_args else None,
        )
        gateway_key = (
            model_path,
            gateway_config["policy"],
            gateway_config["timeout"],
            tuple(gateway_config["extra_args"]) if gateway_config["extra_args"] else None,
            gateway_config.get("log_level"),
            gateway_config.get("log_dir"),
            log_dir,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("WorkerPool has been closed")
            if (
                self._key == key
                and self._gateway_key == gateway_key
                and self._gateway is not None
                and self._zmq_ready(count)
            ):
                logger.info("WorkerPool: reusing ZMQ gateway/worker pair for %s", key)
                return self._gateway

            self._evict_locked()
            gateway = Gateway()
            workers: list[Worker] = []
            try:
                workers = start_workers(
                    model_id=model_id,
                    engine=engine,
                    mode=ConnectionMode.ZMQ,
                    count=count,
                    gpus=gpus,
                    extra_engine_args=extra_engine_args,
                    log_dir=log_dir,
                )
                gateway.start(
                    worker_urls=[w.base_url for w in workers],
                    model_path=model_path,
                    backend=engine,
                    policy=gateway_config["policy"],
                    timeout=gateway_config["timeout"],
                    extra_args=gateway_config["extra_args"],
                    log_level=gateway_config.get("log_level"),
                    log_dir=gateway_config.get("log_dir"),
                )
            except BaseException:
                try:
                    gateway.shutdown()
                finally:
                    stop_workers(workers)
                raise
            self._key, self._workers = key, workers
            self._gateway, self._gateway_key = gateway, gateway_key
            return gateway

    def _zmq_ready(self, count: int) -> bool:
        if (
            self._gateway is None
            or len(self._workers) != count
            or not all(w.is_alive() for w in self._workers)
            or not self._gateway.is_running
        ):
            return False
        try:
            # /health is only gateway liveness. Reuse the startup gate so a
            # live launcher with an unavailable engine is not a cache hit.
            wait_for_workers_ready(self._gateway.base_url, count, timeout=5)
            return True
        except TimeoutError:
            return False

    def discard_zmq(self, gateway: Gateway) -> None:
        """Evict a pair whose class failed, without interrupting that class."""
        with self._lock:
            if self._gateway is gateway:
                self._evict_locked()

    def cleanup(self) -> None:
        """Stop all cached workers. Idempotent; safe to call multiple times."""
        with self._lock:
            self._evict_locked()
            self._closed = True

    def _evict_locked(self) -> None:
        try:
            if self._gateway is not None:
                self._gateway.shutdown()
        finally:
            stop_workers(self._workers)
            self._workers = []
            self._key = None
            self._gateway = None
            self._gateway_key = None


_POOL: WorkerPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> WorkerPool:
    """Return the session-wide worker pool, creating it on first use."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None or _POOL._closed:
            _POOL = WorkerPool()
        return _POOL


def cleanup_pool() -> None:
    """Tear down the session-wide pool if it exists. Called from session-end hook."""
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.cleanup()


# Register the module-level cleanup once at import time. Calling it after
# the pool has already been torn down (by ``pytest_sessionfinish``) is a
# no-op — ``cleanup_pool`` short-circuits when the slot is empty.
atexit.register(cleanup_pool)
