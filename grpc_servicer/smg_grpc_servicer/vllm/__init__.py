"""vLLM gRPC servicers -- VllmEngine proto service and standard health check.

The servicer classes import vLLM and load on first attribute access, so the
engine-free submodules (media_refs, mm_processor, mm_tensors) import without it.
"""

import logging
import os

__all__ = [
    "HELD_RECORDS",
    "METRICS_HOST_ENV",
    "METRICS_PORT_ENV",
    "MULTIPROC_METHOD_ENV",
    "SERVICER_IMPL_ENV",
    "VllmEngineServicer",
    "VllmHealthServicer",
    "attach_vllm_logging",
    "has_handlers",
    "release_held_records",
    "resolve_servicer_impl",
    "serve_rust",
    "settle_logging",
]

# The flag that selects the Rust request path; see `smg_grpc_servicer.vllm.rust`.
SERVICER_IMPL_ENV = "SMG_VLLM_SERVICER_IMPL"
# The Rust servicer's Prometheus endpoint: the fallbacks of --metrics-port and
# --metrics-host (see `smg_grpc_servicer.rust_lifecycle.metrics_bind_address`).
METRICS_PORT_ENV = "SMG_VLLM_SERVICER_METRICS_PORT"
METRICS_HOST_ENV = "SMG_VLLM_SERVICER_METRICS_HOST"
# vLLM's own knob for how it starts its processes; the engine cores launched
# from the servicer process are spawned unless the deployment chose otherwise
# (see `smg_grpc_servicer.vllm.rust.launch_engine_cores`).
MULTIPROC_METHOD_ENV = "VLLM_WORKER_MULTIPROC_METHOD"
# How many of this package's records are held for the handlers that come later
# (see `attach_vllm_logging`): a launch logs a few before vLLM configures
# logging; a process that never configures any keeps no more than this.
HELD_RECORDS = 64

_PACKAGE_LOGGER = "smg_grpc_servicer"


class _HeldRecords(logging.Handler):
    """This package's records from before any handler could print them, in
    order, at most :data:`HELD_RECORDS`; and the package logger's level from
    before the hold, for the release to put back what the hold set. A record
    another handler prints (one configured since the hold began: the root
    logger's, say) is not kept, so the release prints nothing twice."""

    def __init__(self, level_before: int, level_set: int | None) -> None:
        super().__init__()
        self.level_before = level_before
        self.level_set = level_set
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.records) < HELD_RECORDS and not has_handlers():
            self.records.append(record)


def _bootstrap_level() -> int:
    """The level vLLM's handler starts at (``VLLM_LOGGING_LEVEL``, INFO when
    unset): what the package logs at while its records are held."""
    level = logging.getLevelName(os.environ.get("VLLM_LOGGING_LEVEL", "INFO").strip().upper())
    return level if isinstance(level, int) else logging.INFO


def _hold(pkg_logger: logging.Logger) -> None:
    if any(isinstance(handler, _HeldRecords) for handler in pkg_logger.handlers):
        return
    level_set = _bootstrap_level() if pkg_logger.level == logging.NOTSET else None
    pkg_logger.addHandler(_HeldRecords(pkg_logger.level, level_set))
    if level_set is not None:
        pkg_logger.setLevel(level_set)


def _release(pkg_logger: logging.Logger) -> list[logging.LogRecord]:
    held: list[logging.LogRecord] = []
    for handler in list(pkg_logger.handlers):
        if not isinstance(handler, _HeldRecords):
            continue
        pkg_logger.removeHandler(handler)
        if handler.level_set is not None and pkg_logger.level == handler.level_set:
            pkg_logger.setLevel(handler.level_before)
        held.extend(handler.records)
    return held


def _replay(records: list[logging.LogRecord]) -> None:
    """Each held record again, through what its logger reaches now, as a new
    record of the same logger, level, file, line, message and attributes:
    vLLM's record factory adds what its formatter prints, which a record made
    before the factory was installed lacks. The arguments go on the new record
    as they are (a single mapping argument is already unwrapped on the held
    one, and the record constructor would unwrap it again), and the caller's
    ``extra`` attributes come along as the record's own."""
    for held in records:
        logger = logging.getLogger(held.name)
        record = logger.makeRecord(
            held.name,
            held.levelno,
            held.pathname,
            held.lineno,
            held.msg,
            (),
            held.exc_info,
            held.funcName,
            None,
            held.stack_info,
        )
        record.args = held.args
        for key, value in vars(held).items():
            if key not in vars(record):
                setattr(record, key, value)
        logger.handle(record)


def has_handlers() -> bool:
    """Whether a handler other than the hold is in place for this package's
    records: on its own logger (vLLM's, or someone's), or on the root logger
    they propagate to."""
    pkg_logger = logging.getLogger(_PACKAGE_LOGGER)
    return bool(logging.getLogger().handlers) or any(
        not isinstance(handler, _HeldRecords) for handler in pkg_logger.handlers
    )


def release_held_records() -> int:
    """End the hold (see :func:`attach_vllm_logging`) and replay the records it
    kept through what the package's logger reaches now; returns how many. For
    a launch that configured a handler of its own where vLLM configured none."""
    held = _release(logging.getLogger(_PACKAGE_LOGGER))
    _replay(held)
    return len(held)


def settle_logging() -> bool:
    """The launcher's moment to configure logging is past (it runs its serving
    coroutine, or builds the servicer): route through vLLM's handlers when it
    configured them, else end the hold, so what was held, and what the package
    logs from here, goes where the standard library sends it (a handler of the
    launcher's own; its last-resort stderr handler for warnings and errors, as
    without the hold). Returns whether vLLM's handlers are attached."""
    if attach_vllm_logging():
        return True
    release_held_records()
    return False


def attach_vllm_logging() -> bool:
    """Route this package's logs through vLLM's handlers once vLLM configured
    them; returns whether they are attached.

    Until then, where no handler would print them, the package's records are
    held: vLLM 0.31 configures logging after the launcher's argument parse,
    during which the plugin's parse step logs the capture-cap decision (0.28
    through 0.30 configured it at import, before anything of this package
    ran). The first call that finds vLLM's handlers, or a handler someone else
    configured, replays them in order (:func:`release_held_records`); the hold
    ends at the latest when the launcher's moment to configure logging is past
    (:func:`settle_logging`). Held at vLLM's bootstrap level
    (``VLLM_LOGGING_LEVEL``, INFO when unset), at most :data:`HELD_RECORDS`.
    """
    vllm_logger = logging.getLogger("vllm")
    pkg_logger = logging.getLogger(_PACKAGE_LOGGER)
    if vllm_logger.handlers:
        held = _release(pkg_logger)
        pkg_logger.handlers = list(vllm_logger.handlers)
        pkg_logger.setLevel(vllm_logger.level)
        pkg_logger.propagate = False
        _replay(held)
        return True
    if has_handlers():
        _replay(_release(pkg_logger))
    else:
        _hold(pkg_logger)
    return False


def __getattr__(name: str):
    if name in ("resolve_servicer_impl", "serve_rust"):
        from smg_grpc_servicer.vllm import rust

        return getattr(rust, name)
    if name == "VllmEngineServicer":
        from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

        return VllmEngineServicer
    if name == "VllmHealthServicer":
        from smg_grpc_servicer.vllm.health_servicer import VllmHealthServicer

        return VllmHealthServicer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
