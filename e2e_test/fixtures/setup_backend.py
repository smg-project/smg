"""Backend setup fixtures for E2E tests.

One gateway per test class; workers come from the session-scoped pool in
``infra.worker_pool`` so they survive class teardown when the next class
needs the same backend. The pool keys reuse on
``(engine, model_id, mode, worker_type, count)`` and gates reuse on
worker liveness.

PD-disaggregation paths run through the pool too (so it can evict a
stale cached worker holding their GPUs) but the caller owns teardown of
the prefill/decode workers via ``stop_workers``. The function-scoped
``backend_router`` fixture intentionally bypasses the pool entirely.
"""

from __future__ import annotations

import logging
import os
import time

import anthropic
import openai
import pytest
from infra import (
    DEFAULT_MODEL,
    DEFAULT_ROUTER_TIMEOUT,
    DEFAULT_STARTUP_TIMEOUT,
    ENV_MODEL,
    ENV_SKIP_BACKEND_SETUP,
    RUNTIME_LABELS,
    THIRD_PARTY_MODELS,
    ConnectionMode,
    Gateway,
    Runtime,
    WorkerType,
    get_connection_mode_override,
    get_runtime,
    launch_cloud_gateway,
)
from infra.model_specs import get_model_spec
from infra.worker import stop_workers
from infra.worker_pool import get_pool

from .markers import get_marker_kwargs, get_marker_value

logger = logging.getLogger(__name__)

_GW_DEFAULTS = {
    "policy": "round_robin",
    "timeout": DEFAULT_ROUTER_TIMEOUT,
    "extra_args": None,
    "log_level": None,
    "log_dir": None,
}

_WORKER_DEFAULTS = {
    "count": 1,
    "prefill": None,
    "decode": None,
    "gpus": None,
    "extra_engine_args": None,
}

# Track worker startup failures — fail fast after repeated failures
_worker_start_failures: dict[str, int] = {}  # engine -> count
_MAX_WORKER_START_FAILURES = 3  # fail fast after this many failures (matches --reruns 2)

# Engines that speak the direct-ZMQ backend wire in e2e. Pairing ZMQ with any
# other engine can't work, so we reject it up front instead of timing out on a
# worker that never becomes ready.
ZMQ_CAPABLE_ENGINES = frozenset({Runtime.VLLM.value, Runtime.TOKENSPEED.value})


def _validate_connection_mode(connection_mode: ConnectionMode, engine: str) -> None:
    """Reject connection-mode/engine pairings that cannot start.

    Raises ``ValueError`` when a lane selects ZMQ for an engine that does not
    support the direct-ZMQ backend.
    """
    if connection_mode == ConnectionMode.ZMQ and engine not in ZMQ_CAPABLE_ENGINES:
        raise ValueError(
            f"ConnectionMode.ZMQ is only supported for engines "
            f"{sorted(ZMQ_CAPABLE_ENGINES)}, not {engine!r}"
        )


def _start_workers_tracked(**kwargs) -> list:
    """Start workers via the session pool and track failures for fail-fast.

    The pool caches REGULAR workers across class boundaries — caller MUST
    NOT call ``stop_workers`` on those. Non-REGULAR workers (PD
    prefill/decode) skip the cache but still run through the pool so it
    can evict any stale cached worker holding their GPUs; the caller still
    owns teardown of those.
    """
    engine = kwargs.get("engine") or get_runtime()
    try:
        return get_pool().acquire(**kwargs)
    except (TimeoutError, RuntimeError):
        _worker_start_failures[engine] = _worker_start_failures.get(engine, 0) + 1
        raise


def _start_gateway(gateway: Gateway, gateway_config: dict, **mode_kwargs) -> None:
    """Start gateway with mode-specific kwargs and shared config."""
    gateway.start(
        **mode_kwargs,
        policy=gateway_config["policy"],
        timeout=gateway_config["timeout"],
        extra_args=gateway_config["extra_args"],
        log_level=gateway_config.get("log_level"),
        log_dir=gateway_config.get("log_dir"),
    )


def _gateway_readiness_timeout(
    connection_mode: ConnectionMode, model_id: str, base_timeout: float
) -> float:
    """Effective gateway readiness timeout for the given connection mode.

    gRPC/HTTP workers are health-checked (model fully loaded) by the pool
    before the gateway starts, so the gateway only has to connect — the short
    router timeout suffices. ZMQ engines instead spawn and return immediately;
    their model load happens *inside* the gateway's readiness gate, so that gate
    must cover model load too. Use the model's ``startup_timeout`` (what the
    worker gate would have applied), never shrinking an explicitly larger
    gateway timeout.
    """
    if connection_mode != ConnectionMode.ZMQ:
        return base_timeout
    startup_timeout = get_model_spec(model_id).get("startup_timeout", DEFAULT_STARTUP_TIMEOUT)
    return max(base_timeout, startup_timeout)


def _make_openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{gateway.base_url}/v1", api_key="not-used")


# Statuses a fresh gateway returns while it is still wiring up (no worker
# routable yet, tokenizer not registered) rather than rejecting the request.
_NOT_SERVING_YET = frozenset({404, 408, 425, 429, 500, 502, 503, 504})


def _wait_for_serving(
    gateway: Gateway, model_id: str, model_path: str, timeout: float = 180.0
) -> None:
    """Block until the gateway answers a real chat request for ``model_path``.

    PD lanes have returned 404 for a class's first request seconds after the
    gateway reported ready, for reasons the readiness gate does not explain.
    Wait for a request to succeed rather than trusting the gate.
    """
    if "chat" not in get_model_spec(model_id).get("features", []):
        return
    client = _make_openai_client(gateway).with_options(max_retries=0)
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                timeout=min(60.0, remaining),
            )
            return
        except openai.APIStatusError as exc:
            if exc.status_code not in _NOT_SERVING_YET:
                raise
            last_error = exc
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            last_error = exc
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        f"Gateway at {gateway.base_url} did not serve a chat request for "
        f"{model_path} within {timeout}s: {last_error}"
    )


# ---------------------------------------------------------------------------
# Main fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def setup_backend(request: pytest.FixtureRequest):
    """Class-scoped fixture that launches workers + gateway for each test class.

    Backend type is determined by parametrize value via ``request.param``:
      - ``"http"``, ``"grpc"``: Local workers (SGLang, vLLM, or TRT-LLM)
      - ``"pd_http"``, ``"pd_grpc"``: PD disaggregation workers
      - ``"openai"``, ``"xai"``, ``"anthropic"``: Cloud backends (no workers)

    Configuration via markers:
      - ``@pytest.mark.model("model-id")``: Override default model
      - ``@pytest.mark.workers(count=1)``: Number of regular workers
      - ``@pytest.mark.workers(gpus=2, extra_engine_args=[...])``: Per-worker
        GPU count and extra engine CLI args (local workers only)
      - ``@pytest.mark.workers(prefill=1, decode=1)``: PD worker counts
      - ``@pytest.mark.gateway(policy=..., timeout=..., extra_args=...)``: Gateway config

    Returns:
        Tuple of ``(backend_name, model_path, client, gateway)``
    """
    raw_param = request.param
    if isinstance(raw_param, tuple):
        backend_name, epd_counts = raw_param
    else:
        backend_name, epd_counts = raw_param, None

    if os.environ.get(ENV_SKIP_BACKEND_SETUP, "").lower() in ("1", "true", "yes"):
        pytest.skip(f"{ENV_SKIP_BACKEND_SETUP} is set")

    model_id = get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    gateway_config = get_marker_kwargs(request, "gateway", defaults=_GW_DEFAULTS)

    # Cloud backends (no local workers)
    if backend_name in THIRD_PARTY_MODELS:
        yield from _setup_cloud(backend_name, request, gateway_config)
        return

    # Local backends
    is_epd = backend_name.startswith("epd_")
    is_pd = backend_name.startswith("pd_")
    protocol = backend_name.replace("epd_", "").replace("pd_", "")
    connection_mode = ConnectionMode(protocol)
    # A lane can override the local wire (e.g. run grpc/http cases over ZMQ);
    # PD/EPD keep their own wire since they are excluded from those lanes.
    mode_override = get_connection_mode_override()
    if mode_override is not None and not is_pd and not is_epd:
        connection_mode = mode_override
    engine = get_runtime()
    _validate_connection_mode(connection_mode, engine)
    model_path = get_model_spec(model_id)["model"]
    workers_config = get_marker_kwargs(request, "workers", defaults=_WORKER_DEFAULTS)
    log_dir = os.environ.get("E2E_LOG_DIR") or gateway_config.get("log_dir")

    fail_count = _worker_start_failures.get(engine, 0)
    if fail_count >= _MAX_WORKER_START_FAILURES:
        pytest.exit(
            f"Engine {engine} failed to start workers {fail_count} times — aborting test session",
            returncode=1,
        )

    gateway = Gateway()
    try:
        if is_epd:
            yield from _setup_epd(
                model_id,
                model_path,
                engine,
                connection_mode,
                epd_counts,
                gateway_config,
                gateway,
                log_dir,
            )
        elif is_pd:
            yield from _setup_pd(
                model_id,
                model_path,
                engine,
                connection_mode,
                workers_config,
                gateway_config,
                gateway,
                log_dir,
            )
        else:
            yield from _setup_local(
                model_id,
                model_path,
                engine,
                connection_mode,
                workers_config,
                gateway_config,
                gateway,
                backend_name,
                log_dir,
            )
    except Exception:
        gateway.shutdown()
        raise


# ---------------------------------------------------------------------------
# Local (non-PD) backend
# ---------------------------------------------------------------------------


def _setup_local(
    model_id,
    model_path,
    engine,
    connection_mode,
    workers_config,
    gateway_config,
    gateway,
    backend_name,
    log_dir,
):
    """Launch regular workers + gateway, yield result tuple, tear down.

    Workers are acquired from the session-scoped pool so they survive class
    teardown when the next class needs the same backend. The gateway is
    per-class (its config can differ across classes) and is torn down here.
    """
    num_workers = workers_config.get("count") or 1
    logger.info("Starting %s backend: model=%s, workers=%d", backend_name, model_id, num_workers)

    workers = _start_workers_tracked(
        model_id=model_id,
        engine=engine,
        mode=connection_mode,
        count=num_workers,
        log_dir=log_dir,
        gpus=workers_config.get("gpus"),
        extra_engine_args=workers_config.get("extra_engine_args"),
    )
    # ZMQ engines dial this gateway's handshake sockets, so they cannot be
    # reused by a later class's gateway — the pool starts them fresh and the
    # caller owns their teardown (like the PD path). gRPC/HTTP workers stay
    # in the pool and outlive the gateway.
    is_zmq = connection_mode == ConnectionMode.ZMQ
    # ZMQ engines load the model inside the gateway's readiness gate (the worker
    # spawn returned immediately), so that gate must cover model load.
    gateway_config = {
        **gateway_config,
        "timeout": _gateway_readiness_timeout(connection_mode, model_id, gateway_config["timeout"]),
    }
    try:
        _start_gateway(
            gateway,
            gateway_config,
            worker_urls=[w.base_url for w in workers],
            model_path=model_path,
            backend=engine if is_zmq else None,
        )
        logger.info("%s backend ready at %s", backend_name, gateway.base_url)
        yield backend_name, model_path, _make_openai_client(gateway), gateway
    finally:
        logger.info(
            "Tearing down %s backend (%s)",
            backend_name,
            "stopping ZMQ workers" if is_zmq else "workers stay in pool",
        )
        gateway.shutdown()
        if is_zmq:
            stop_workers(workers)


# ---------------------------------------------------------------------------
# PD disaggregation backend
# ---------------------------------------------------------------------------


def _setup_pd(
    model_id,
    model_path,
    engine,
    connection_mode,
    workers_config,
    gateway_config,
    gateway,
    log_dir,
):
    """Launch prefill + decode workers + PD gateway, yield, tear down."""
    spec = get_model_spec(model_id)
    num_prefill = workers_config.get("prefill") or 1
    num_decode = workers_config.get("decode") or 1
    backend_name = f"pd_{connection_mode.value}"
    runtime_label = RUNTIME_LABELS.get(engine, engine)

    logger.info(
        "Starting %s PD backend: model=%s, %d prefill + %d decode",
        runtime_label,
        model_id,
        num_prefill,
        num_decode,
    )

    all_workers: list = []
    try:
        prefill_workers = _start_workers_tracked(
            model_id=model_id,
            engine=engine,
            mode=connection_mode,
            count=num_prefill,
            worker_type=WorkerType.PREFILL,
            log_dir=log_dir,
        )
        all_workers.extend(prefill_workers)

        # Decode workers start on GPUs after prefill workers
        decode_gpu_offset = num_prefill * spec.get("tp", 1)
        decode_workers = _start_workers_tracked(
            model_id=model_id,
            engine=engine,
            mode=connection_mode,
            count=num_decode,
            worker_type=WorkerType.DECODE,
            log_dir=log_dir,
            gpu_offset=decode_gpu_offset,
        )
        all_workers.extend(decode_workers)

        _start_gateway(
            gateway,
            gateway_config,
            prefill_workers=prefill_workers,
            decode_workers=decode_workers,
        )
        _wait_for_serving(gateway, model_id, model_path)
        logger.info("%s PD backend ready at %s", runtime_label, gateway.base_url)
        yield backend_name, model_path, _make_openai_client(gateway), gateway
    finally:
        logger.info("Tearing down %s PD backend", runtime_label)
        gateway.shutdown()
        stop_workers(all_workers)


# ---------------------------------------------------------------------------
# EPD (encode-prefill-decode) disaggregation backend
# ---------------------------------------------------------------------------


def _setup_epd(
    model_id,
    model_path,
    engine,
    connection_mode,
    epd_counts,
    gateway_config,
    gateway,
    log_dir,
):
    """Launch encode + prefill + decode workers + EPD gateway, yield, tear down.

    ``epd_counts`` is ``(n_encode, n_prefill, n_decode)``. Every worker is tp=1
    (one GPU); GPUs are assigned sequentially E -> P -> D.
    """
    if not epd_counts or len(epd_counts) != 3:
        raise ValueError("epd_grpc backend requires a (n_encode, n_prefill, n_decode) param")
    n_encode, n_prefill, n_decode = epd_counts
    spec = get_model_spec(model_id)
    tp = spec.get("tp", 1)
    backend_name = f"epd_{connection_mode.value}"
    runtime_label = RUNTIME_LABELS.get(engine, engine)

    logger.info(
        "Starting %s EPD backend: model=%s, %de + %dp + %dd",
        runtime_label,
        model_id,
        n_encode,
        n_prefill,
        n_decode,
    )

    all_workers: list = []
    try:
        encode_workers = _start_workers_tracked(
            model_id=model_id,
            engine=engine,
            mode=connection_mode,
            count=n_encode,
            worker_type=WorkerType.ENCODE,
            log_dir=log_dir,
            gpu_offset=0,
            gpus=1,  # vision tower runs on one GPU regardless of LM tp
        )
        all_workers.extend(encode_workers)

        prefill_workers = _start_workers_tracked(
            model_id=model_id,
            engine=engine,
            mode=connection_mode,
            count=n_prefill,
            worker_type=WorkerType.PREFILL,
            log_dir=log_dir,
            gpu_offset=n_encode,
        )
        all_workers.extend(prefill_workers)

        decode_workers = _start_workers_tracked(
            model_id=model_id,
            engine=engine,
            mode=connection_mode,
            count=n_decode,
            worker_type=WorkerType.DECODE,
            log_dir=log_dir,
            gpu_offset=n_encode + n_prefill * tp,
        )
        all_workers.extend(decode_workers)

        _start_gateway(
            gateway,
            gateway_config,
            encode_workers=encode_workers,
            prefill_workers=prefill_workers,
            decode_workers=decode_workers,
        )
        logger.info("%s EPD backend ready at %s", runtime_label, gateway.base_url)
        yield backend_name, model_path, _make_openai_client(gateway), gateway
    finally:
        logger.info("Tearing down %s EPD backend", runtime_label)
        gateway.shutdown()
        stop_workers(all_workers)


# ---------------------------------------------------------------------------
# Cloud backend
# ---------------------------------------------------------------------------


def _setup_cloud(backend_name, request, gateway_config):
    """Launch cloud gateway (no local workers), yield result tuple, tear down."""
    cfg = THIRD_PARTY_MODELS[backend_name]
    api_key_env = cfg.get("api_key_env")

    if api_key_env and not os.environ.get(api_key_env):
        pytest.fail(f"{api_key_env} not set for {backend_name} tests")

    storage_backend = get_marker_value(request, "storage", default="memory")

    logger.info("Launching cloud backend: %s (storage=%s)", backend_name, storage_backend)
    gateway = launch_cloud_gateway(
        backend_name,
        history_backend=storage_backend,
        extra_args=gateway_config.get("extra_args"),
    )

    api_key = os.environ.get(api_key_env) if api_key_env else "not-used"
    model_path = cfg["model"]

    client: openai.OpenAI | anthropic.Anthropic
    if cfg.get("client_type") == "anthropic":
        client = anthropic.Anthropic(base_url=gateway.base_url, api_key=api_key)
    else:
        client = openai.OpenAI(base_url=f"{gateway.base_url}/v1", api_key=api_key)

    try:
        yield backend_name, model_path, client, gateway
    finally:
        logger.info("Tearing down cloud backend: %s", backend_name)
        gateway.shutdown()


# ---------------------------------------------------------------------------
# Per-test gateway fixture (isolated router state)
# ---------------------------------------------------------------------------


@pytest.fixture
def backend_router(request: pytest.FixtureRequest):
    """Function-scoped fixture that launches a fresh gateway per test.

    A new gateway is started for each test function; the worker comes
    from the session-scoped pool (so the GPU isn't fought over with
    class-scope tests). Use when tests need isolated router state.

    Usage::

        @pytest.mark.parametrize("backend_router", ["grpc", "http"], indirect=True)
        def test_router_state(backend_router):
            gateway = backend_router
    """
    backend_name = request.param
    model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)
    connection_mode = ConnectionMode(backend_name)
    mode_override = get_connection_mode_override()
    if mode_override is not None:
        connection_mode = mode_override
    engine = get_runtime()
    _validate_connection_mode(connection_mode, engine)
    model_path = get_model_spec(model_id)["model"]
    is_zmq = connection_mode == ConnectionMode.ZMQ

    # Route through the pool so we evict any cached class-scope worker
    # holding the GPUs we need. The pool retains ownership of gRPC/HTTP
    # workers; ZMQ engines are bound to this gateway, so we stop them here.
    workers = get_pool().acquire(
        model_id=model_id,
        engine=engine,
        mode=connection_mode,
        count=1,
    )
    gateway = Gateway()
    try:
        gateway.start(
            worker_urls=[w.base_url for w in workers],
            model_path=model_path,
            backend=engine if is_zmq else None,
            # ZMQ loads the model inside the gateway's readiness gate; cover it.
            timeout=_gateway_readiness_timeout(connection_mode, model_id, DEFAULT_ROUTER_TIMEOUT),
        )
        yield gateway
    finally:
        gateway.shutdown()
        if is_zmq:
            stop_workers(workers)
