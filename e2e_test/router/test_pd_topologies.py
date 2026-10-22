"""Prefill/decode disaggregation under every worker topology a 4-GPU node allows.

One class sweeps the topologies 1p1d, 1p2d, 2p1d, 1p3d, 3p1d and 2p2d, two
asymmetric-tp pairs (1p2d-ptp2-dtp1, 2p1d-ptp1-dtp2) and the HTTP PD router's
1p1d and 2p2d. The counts, and the per-pool tp when the pools differ, ride in
the ``setup_backend`` param because the fixture is class-scoped, and every
test runs once per topology. The gateway logs each
selected pair at debug level, so requests are attributed to workers from the
router log instead of guessed.

What a user of PD mode hits, in the order they hit it:

- registration: ``/workers`` lists every worker with its role, all healthy
- placement: every prefill and every decode worker takes traffic
- transfer: a long prompt's context reaches decode; concurrent requests do
  not see each other's KV; a second turn still sees the first
- streaming: a stream completes with usage; an abandoned stream frees the
  engine slots on both pools instead of generating to its limit
- failure: a pool loses one worker and traffic keeps flowing; a pool loses its
  only worker and the client gets a prompt error instead of a hang; the
  worker comes back and rejoins rotation; a decode engine dies under live
  streams and each stream ends with an error frame, never silently, while
  its peers keep serving and its restart on the same GPU rejoins
- operations: a PD fleet assembled at runtime through the worker API in IGW
  mode routes through prefill and decode

TokenSpeed runs Qwen3.5-9B, the model its disaggregation path is proven on;
SGLang and vLLM run Llama-3.1-8B.

A MoRI-IO lane (``E2E_VLLM_KV_BACKEND=moriio``, ``E2E_VLLM_MORIIO_MODE`` read
or write) runs every topology on SMG's HTTP PD router, the one that speaks the
MoRI-IO handoff, so every row moves real KV. It reads idleness from the
engines' own metrics (HTTP workers send no load reports), checks that decode
computed none of a long prompt, and expects SMG to refuse ``n>1``.

Usage:
    E2E_RUNTIME=tokenspeed pytest e2e_test/router/test_pd_topologies.py -v -k 2p2d
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from infra import ConnectionMode, Gateway, WorkerType, cleanup_pool, start_workers, stop_workers
from infra.constants import (
    DEFAULT_STARTUP_TIMEOUT,
    get_gpu_vendor,
    get_runtime,
    is_sglang,
    vllm_kv_backend,
    vllm_moriio_mode,
)
from infra.model_specs import get_model_spec
from infra.pd_logs import (
    LOG_FLUSH_TIMEOUT_S,
    assert_worker_logs_captured,
    read_logs,
    worker_log_dir,
)
from infra.streams import SseChatEnd, read_sse_chat

logger = logging.getLogger(__name__)

# Router logs land here via the gateway marker (rolling files named smg.YYYY-MM-DD)
_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-pd-topologies-{os.getpid()}"
PAIR_MARKER = "Selected PD pair"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PAIR_RE = re.compile(r"prefill=(\S+)\s+decode=(\S+)")

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_MODEL_BY_ENGINE = {"tokenspeed": "Qwen/Qwen3.5-9B"}
# A restart reloads the model on the same GPUs: give it the budget the model
# declares for its first load, not Worker.start()'s default.
_RESTART_TIMEOUT = get_model_spec(_MODEL_BY_ENGINE.get(get_runtime(), _MODEL)).get(
    "startup_timeout", DEFAULT_STARTUP_TIMEOUT
)
_HEALTH_ARGS = [
    "--health-check-interval-secs",
    "1",
    "--health-check-timeout-secs",
    "2",
    "--health-failure-threshold",
    "1",
    "--health-success-threshold",
    "1",
]


def _moriio_lane() -> bool:
    """vLLM PD with KV over MoRI-IO, which setup_backend runs on the HTTP PD router."""
    return get_runtime() == "vllm" and vllm_kv_backend() == "moriio"


def _moriio_holds_memory_of(role: str) -> bool:
    """Whether the other role keeps a stopped ``role`` worker's GPU memory in use.

    On one node the role that moves the KV (decode in READ, prefill in WRITE)
    keeps the memory of each worker it moved KV with until it exits itself,
    over RDMA as over XGMI, so a worker restarted on that GPU fails vLLM's
    startup memory check.
    """
    if not _moriio_lane():
        return False
    return role == ("prefill" if vllm_moriio_mode() == "read" else "decode")


_MORIIO_HELD_MEMORY = (
    "the other MoRI-IO role keeps this worker's KV cache until it exits, "
    "so a restart on the same GPU runs out of memory (vllm-project/vllm#60425)"
)


def _nixl_decodes_keep_a_dead_prefills_memory(role: str) -> bool:
    """Whether the decodes keep a stopped sole ``role`` worker's KV region until they serve again.

    A vLLM decode over NIXL maps the prefill's KV region it pulls from and
    forgets an idle prefill (``engine_ttl``) only while it handles a request
    (0.31.0; 0.27.1 freed it when the prefill exited). A pool of two keeps
    serving through its survivor, so that restart passes; the sole prefill's
    decodes idle behind the router's 503, nothing makes them forget it, and
    its restart on the same GPU fails vLLM's startup memory check. Seen on
    CUDA only: the ROCm lane's decodes release the region when the prefill
    exits and its restart passes, so that lane keeps the case. Lift the skip
    when a release frees a dead peer's region without traffic (re-check it
    at every engine pin move).
    """
    return (
        role == "prefill"
        and get_runtime() == "vllm"
        and vllm_kv_backend() == "nixl"
        and get_gpu_vendor() == "nvidia"
    )


_NIXL_HELD_MEMORY = (
    "a vLLM decode over NIXL forgets a dead prefill only while it serves a request, and none "
    "reaches it while the prefill pool is empty, so the sole prefill restarted on the same GPU "
    "fails the engine's startup memory check (vLLM 0.31.0 on CUDA; the ROCm lane releases the "
    "region and keeps this case; lift the skip when a release frees a dead peer's region "
    "without traffic)"
)


_GATEWAY_ARGS = [
    "--prefill-policy",
    "round_robin",
    "--decode-policy",
    "round_robin",
    "--load-monitor-interval",
    "2",
    *_HEALTH_ARGS,
]
_TOPOLOGIES = (
    [
        pytest.param(("pd_grpc", (p, d)), id=f"{p}p{d}d")
        for p, d in [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
    ]
    + [
        # Asymmetric tensor parallelism: the KV layout changes across the
        # handoff. A prefill wider than its decodes and the reverse both fit
        # four GPUs.
        pytest.param(
            ("pd_grpc", (p, d, ptp, dtp)),
            id=f"{p}p{d}d-ptp{ptp}-dtp{dtp}",
            marks=pytest.mark.skipif(
                _moriio_lane(),
                reason="vLLM's MoRI-IO connector pairs unequal TP sizes only when every "
                "rank holds all KV heads (vllm-project/vllm#46332); the model's 8 are split",
            ),
        )
        for p, d, ptp, dtp in [(1, 2, 2, 1), (2, 1, 1, 2)]
    ]
    + [
        # The HTTP PD router is its own code path: pairing and error answers
        # on both engines, the KV handoff on SGLang. A vLLM worker registered by
        # URL carries no kv_connector, so the router passes the request
        # through and decode recomputes the prompt; vLLM's handoff is covered
        # by the gRPC rows, where the worker reports its connector. The
        # TokenSpeed e2e worker has no HTTP frontend.
        pytest.param(
            ("pd_http", (p, d)),
            id=f"{p}p{d}d-http",
            marks=pytest.mark.skip_for_runtime(
                "tokenspeed", reason="the TokenSpeed e2e worker has no HTTP frontend"
            ),
        )
        for p, d in [(1, 1), (2, 2)]
    ]
)
# A decode window a modest burst overruns. The gateway's admission gate, bounded
# by the window the engine reports, is what keeps the excess out of the engine,
# where the prefill's bootstrap deadline would expire on merely queued requests.
_WINDOW = 4
_WINDOW_ARGS = {
    "sglang": ["--max-running-requests", str(_WINDOW)],
    "vllm": ["--max-num-seqs", str(_WINDOW)],
    "tokenspeed": ["--max-num-seqs", str(_WINDOW)],
}.get(get_runtime(), [])
# What an SGLang-lineage prefill logs when it gave up waiting for the decode.
_BOOTSTRAP_TIMEOUT_MARKERS = ("timed out when bootstrapping",)
_WORDS = [
    "apricot",
    "bramble",
    "cobalt",
    "dahlia",
    "ember",
    "fjord",
    "glacier",
    "harbor",
    "iris",
    "juniper",
    "kestrel",
    "lagoon",
    "meadow",
    "nectar",
    "orchid",
    "pebble",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pairs_logged() -> list[tuple[str, str]]:
    """(prefill, decode) URL pairs the gateway has logged so far, in order."""
    pairs: list[tuple[str, str]] = []
    for raw in read_logs(_LOG_DIR, "smg*").splitlines():
        if PAIR_MARKER not in raw:
            continue
        match = _PAIR_RE.search(_ANSI.sub("", raw))
        if match:
            pairs.append((match.group(1), match.group(2)))
    return pairs


def _wait_for_pairs(minimum: int, timeout: float = LOG_FLUSH_TIMEOUT_S) -> list[tuple[str, str]]:
    deadline = time.monotonic() + timeout
    pairs = _pairs_logged()
    while len(pairs) < minimum and time.monotonic() < deadline:
        time.sleep(0.5)
        pairs = _pairs_logged()
    assert len(pairs) >= minimum, (
        f"router logged {len(pairs)} PD pair selections, expected at least {minimum}; "
        f"is --log-level debug reaching {_LOG_DIR}/smg*?"
    )
    return pairs


def _log_sizes(log_dir: Path) -> dict[Path, int]:
    """Byte size of every worker log now, to read only what is appended later."""
    return {p: p.stat().st_size for p in log_dir.glob("worker-*.log") if p.is_file()}


def _logs_since(log_dir: Path, sizes: dict[Path, int]) -> str:
    """What each worker log gained since ``sizes`` was taken, per file."""
    parts: list[str] = []
    for path in sorted(log_dir.glob("worker-*.log")):
        if not path.is_file():
            continue
        with path.open("rb") as fh:
            # A restart moves the previous log aside and reopens the path at
            # 0 bytes: a stale offset past EOF would silently read nothing, so
            # fall back to the whole file.
            start = sizes.get(path, 0)
            fh.seek(start if start <= path.stat().st_size else 0)
            parts.append(fh.read().decode("utf-8", errors="replace"))
    return "\n".join(parts)


def _fleet_logs(sizes: dict[Path, int], gateway: Gateway) -> str:
    """The captured log files of this gateway's own workers, named by port."""
    workers = gateway.prefill_workers + gateway.decode_workers
    ports = {str(urlparse(w.base_url).port) for w in workers}
    return "\n".join(str(p) for p in sorted(sizes) if p.stem.rsplit("_", 1)[-1] in ports)


def _vllm_transport_installed(*packages: str) -> bool:
    """Whether every KV transport package is importable from this interpreter."""
    return all(importlib.util.find_spec(package) is not None for package in packages)


def _vllm_transports_requested() -> set[str]:
    """The vLLM transports the lane asked to have installed."""
    names = [os.environ.get("E2E_VLLM_KV_BACKEND", ""), os.environ.get("E2E_KV_BACKEND", "")]
    names += os.environ.get("E2E_VLLM_EXTRA_KV_BACKENDS", "").split(",")
    return {name.strip().lower() for name in names if name.strip()}


# Skip only where the lane never asked for both transports; where it did, a
# missing one fails at worker startup instead of vanishing as a skip.
_NEEDS_BOTH_VLLM_TRANSPORTS = pytest.mark.skipif(
    not _vllm_transport_installed("nixl", "mooncake")
    and not {"nixl", "mooncake"} <= _vllm_transports_requested(),
    reason="a fleet that mixes NIXL and Mooncake needs both transfer engines installed",
)


def _pairing_keys(gateway: Gateway) -> dict[str, str]:
    """Each PD worker's `pd_pairing` key as the gateway reports it on /workers."""
    return {
        worker.url: worker.metadata["pd_pairing"]
        for worker in gateway.list_workers(strict=True)
        if worker.metadata.get("pd_pairing")
    }


def _wait_for_healthy_workers(gateway: Gateway, count: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        healthy = [w for w in gateway.list_workers() if w.status == "healthy"]
        if len(healthy) >= count:
            return
        time.sleep(1.0)
    raise AssertionError(f"fewer than {count} healthy workers after {timeout}s")


def _workers_by_role(gateway: Gateway) -> dict[str, list]:
    by_role: dict[str, list] = {}
    for worker in gateway.list_workers(strict=True):
        by_role.setdefault(str(worker.metadata.get("worker_type")), []).append(worker)
    return by_role


def _status_of(gateway: Gateway, url: str) -> str | None:
    for worker in gateway.list_workers(strict=True):
        if worker.url == url:
            return worker.status
    return None


def _wait_for_status(gateway: Gateway, url: str, wanted: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    seen: str | None = None
    while time.monotonic() < deadline:
        seen = _status_of(gateway, url)
        if seen == wanted:
            return
        time.sleep(0.5)
    pytest.fail(f"worker {url} never became {wanted} within {timeout:.0f}s (last: {seen})")


def _wait_for_removal(gateway: Gateway, url: str, timeout: float) -> None:
    """A removal drains in-flight work first; the worker leaves /workers after that."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _status_of(gateway, url) is None:
            return
        time.sleep(0.5)
    pytest.fail(f"worker {url} still listed {timeout:.0f}s after its removal was accepted")


def _answer_text(message: dict) -> str:
    """Content plus any reasoning text: thinking models may answer in either."""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return f"{content}\n{reasoning}".strip()


def _ask(client, model: str, content: str, *, max_tokens: int = 256, **kwargs) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=max_tokens,
        timeout=120.0,
        **kwargs,
    )
    return response.choices[0].message.model_dump()


def _raw_chat(gateway: Gateway, model: str, content: str, *, timeout: float) -> httpx.Response:
    return httpx.post(
        f"{gateway.base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout=timeout,
    )


def _error_code(resp: httpx.Response) -> str | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    error = body.get("error", body)
    return error.get("code") if isinstance(error, dict) else None


def _wait_until_served(gateway: Gateway, model: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "no attempt"
    while time.monotonic() < deadline:
        try:
            resp = _raw_chat(gateway, model, "Say hello.", timeout=60.0)
        except httpx.HTTPError as exc:
            last = repr(exc)
        else:
            if resp.status_code == 200:
                return
            last = f"{resp.status_code} {resp.text[:200]}"
        time.sleep(1.0)
    pytest.fail(f"gateway did not serve within {timeout:.0f}s; last: {last}")


def _launched_transport(worker) -> str:
    """A worker's KV transport as the gateway names it in its pairing key."""
    backend = worker.effective_kv_backend()
    return f"moriio-{vllm_moriio_mode()}" if backend == "moriio" else backend


def _metric_samples(text: str, name: str, **labels: str) -> list[float]:
    """Values of a Prometheus metric's samples that carry ``labels``."""
    samples = []
    for line in text.splitlines():
        rest = line[len(name) :] if line.startswith(name) else None
        if rest is None or (rest and rest[0] not in " {"):
            continue  # another metric, or a longer name with this prefix
        if all(f'{k}="{v}"' in rest for k, v in labels.items()):
            samples.append(float(line.rsplit(" ", 1)[1]))
    return samples


def _worker_metric(worker, name: str, **labels: str) -> float:
    """Sum of one vLLM worker's samples of a metric that carry ``labels``, from
    its /metrics. vLLM exports the series this file reads from start-up, so a
    worker without one fails the test instead of reading as zero."""
    response = httpx.get(f"{worker.http_url}/metrics", timeout=5.0)
    response.raise_for_status()
    samples = _metric_samples(response.text, name, **labels)
    where = f" with {labels}" if labels else ""
    assert samples, f"{worker.base_url} exports no {name} sample{where}"
    return sum(samples)


def _engines_idle(gateway: Gateway) -> tuple[bool, list[dict]]:
    """The idle check from each vLLM worker's own /metrics, for workers without load reports."""
    states = [
        {
            "worker": w.base_url,
            "num_running_reqs": _worker_metric(w, "vllm:num_requests_running"),
            "num_waiting_reqs": _worker_metric(w, "vllm:num_requests_waiting"),
        }
        for w in gateway.prefill_workers + gateway.decode_workers
    ]
    busy = [s for s in states if s["num_running_reqs"] + s["num_waiting_reqs"] > 0]
    return (not busy, states)


def _fleet_idle(gateway: Gateway) -> tuple[bool, list[dict]]:
    if _moriio_lane():
        return _engines_idle(gateway)
    resp = httpx.get(f"{gateway.base_url}/loads", timeout=5.0)
    assert resp.status_code == 200, resp.text
    loads = resp.json().get("loads", [])
    # /loads omits workers the monitor has no report for; an unreported worker
    # is not evidence of idleness.
    reported = {e.get("worker") for e in loads}
    expected = {w.base_url for w in gateway.prefill_workers + gateway.decode_workers}
    if not expected <= reported:
        return (False, loads)
    busy = [e for e in loads if e.get("num_running_reqs", 0) + e.get("num_waiting_reqs", 0) > 0]
    return (not busy, loads)


def _require_load_reports(backend: str) -> None:
    """The idle check reads /loads; HTTP SGLang workers report none without metrics.

    A MoRI-IO lane reads its engines' own metrics instead (see ``_fleet_idle``).
    """
    if backend == "pd_http" and not _moriio_lane():
        pytest.skip("load reports need gRPC workers")


def _assert_fleet_idle_within(gateway: Gateway, timeout: float) -> None:
    started = time.monotonic()
    deadline = started + timeout
    idle, loads = _fleet_idle(gateway)
    while not idle and time.monotonic() < deadline:
        time.sleep(1.0)
        idle, loads = _fleet_idle(gateway)
    # A floor on how long the engines kept working after the clients left:
    # the clock starts once the callers joined their threads, and /loads is
    # a 2 s-interval report, so only differences well above that mean much.
    if idle:
        logger.info("fleet idle within %.1fs of the clients leaving", time.monotonic() - started)
    assert idle, f"engines still hold work after {timeout:.0f}s: {loads}"


def _run_all(fns: list[Callable[[], None]], timeout: float) -> list[BaseException]:
    errors: list[BaseException] = []

    def _wrap(fn: Callable[[], None]) -> None:
        try:
            fn()
        except BaseException as exc:  # collected for the caller's assertion
            errors.append(exc)

    threads = [threading.Thread(target=_wrap, args=(fn,), daemon=True) for fn in fns]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
        if t.is_alive():
            errors.append(TimeoutError(f"a request was still running {timeout:.0f}s in"))
    return errors


@dataclass
class _StreamOutcome:
    """How one streamed chat completion ended: the response status, the SSE
    stream's end (see :class:`SseChatEnd`), or the exception that cut it."""

    status: int | None = None
    end: SseChatEnd = field(default_factory=SseChatEnd)
    exception: str | None = None


def _stream_chat(
    gateway: Gateway, model: str, *, max_tokens: int, first_token: threading.Event
) -> _StreamOutcome:
    """Stream one chat completion to its end, ``first_token`` set at its first
    chunk. A response that is not a stream (a status other than 200) is kept
    as an error with its body."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a very long story about the sea."}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }
    outcome = _StreamOutcome()
    try:
        with httpx.stream(
            "POST", f"{gateway.base_url}/v1/chat/completions", json=body, timeout=120.0
        ) as resp:
            outcome.status = resp.status_code
            if resp.status_code != 200:
                outcome.end.error = {"http_status": resp.status_code, "body": resp.read()[:300]}
                return outcome
            outcome.end = read_sse_chat(resp.iter_lines(), on_first_chunk=first_token.set)
    except Exception as exc:  # reported in the outcome, judged by the test
        outcome.exception = repr(exc)
    return outcome


def _long_prompt(secret: str) -> str:
    filler = (
        "The harbor office logs every vessel by name, tonnage and berth, and the "
        "clerk reads the tide tables twice before the evening shift begins. "
    )
    # ~3k tokens: enough KV blocks that a transfer that drops blocks shows up.
    return (
        f"Instruction: the access code is {secret}. Remember it.\n\n"
        + filler * 120
        + "\n\nQuestion: what is the access code? Reply with the number only."
    )


# ---------------------------------------------------------------------------
# the topology sweep
# ---------------------------------------------------------------------------


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model(_MODEL, tokenspeed=_MODEL_BY_ENGINE["tokenspeed"])
@pytest.mark.workers(parallel_start=True)
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR), extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize("setup_backend", _TOPOLOGIES, indirect=True)
class TestPDTopology:
    """Every test below runs once per prefill/decode topology."""

    # -- registration ------------------------------------------------------

    def test_workers_registered_by_role(self, setup_backend):
        mode, model, _, gateway = setup_backend
        expected = {
            "prefill": {w.base_url for w in gateway.prefill_workers},
            "decode": {w.base_url for w in gateway.decode_workers},
        }

        by_role = _workers_by_role(gateway)
        listed = {role: {w.url for w in workers} for role, workers in by_role.items()}

        assert listed == expected, f"/workers roles {listed} != launched {expected}"
        unhealthy = [w.url for ws in by_role.values() for w in ws if w.status != "healthy"]
        assert not unhealthy, f"unhealthy workers after startup: {unhealthy}"
        assert any(m.get("id") for m in gateway.list_models()), "no model listed"

        # Every worker reports the KV transport the lane launched it with: a row
        # that names a transport its workers never took sweeps the wrong thing
        # (#2498). gRPC discovery reads the engine's own server args, so the
        # transport is known there; HTTP discovery's curated server_info does
        # not carry it and reads as "?". TokenSpeed is not pinned by this: it
        # only moves KV over Mooncake, so both sides are that constant.
        keys = _pairing_keys(gateway)
        logger.info("pairing keys: %s", keys)
        workers = gateway.prefill_workers + gateway.decode_workers
        reported = {w.base_url: keys.get(w.base_url, "?/?").split("/")[1] for w in workers}
        launched = {w.base_url: _launched_transport(w) for w in workers}
        wrong = {
            u: (reported[u], launched[u]) for u in launched if reported[u] not in ("?", launched[u])
        }
        assert not wrong, (
            f"workers report a different KV transport than they were launched with: {wrong}"
        )
        if mode == "pd_grpc":
            unknown = [u for u, t in reported.items() if t == "?"]
            assert not unknown, f"gRPC workers without a known KV transport on /workers: {unknown}"

    # -- placement ---------------------------------------------------------

    def test_every_worker_takes_traffic(self, setup_backend):
        _, model, client, gateway = setup_backend
        prefills = {w.base_url for w in gateway.prefill_workers}
        decodes = {w.base_url for w in gateway.decode_workers}
        before = len(_pairs_logged())
        count = 4 * max(len(prefills), len(decodes))

        for i in range(count):
            _ask(client, model, f"Reply with the number {i}.", max_tokens=8)

        pairs = _wait_for_pairs(before + count)[before:]
        used_prefill = {p for p, _ in pairs}
        used_decode = {d for _, d in pairs}
        assert used_prefill == prefills, f"idle prefill workers: {prefills - used_prefill}"
        assert used_decode == decodes, f"idle decode workers: {decodes - used_decode}"
        assert used_prefill.isdisjoint(decodes), "a decode worker served as prefill"

    # -- transfer ----------------------------------------------------------

    def test_concurrent_requests_do_not_alias(self, setup_backend):
        _, model, client, gateway = setup_backend
        answers: dict[str, str] = {}

        def _echo(word: str) -> Callable[[], None]:
            def _run() -> None:
                message = _ask(
                    client, model, f"Repeat exactly this one word and nothing else: {word}"
                )
                answers[word] = _answer_text(message)

            return _run

        errors = _run_all([_echo(w) for w in _WORDS], timeout=300)

        assert not errors, f"requests failed: {errors[:3]}"
        assert set(answers) == set(_WORDS), f"missing answers for {set(_WORDS) - set(answers)}"
        for word, text in answers.items():
            assert word in text.lower(), f"answer for {word!r} lost it: {text[:120]!r}"
            content = (text.split("\n", 1)[0]).lower()
            foreign = [w for w in _WORDS if w != word and w in content]
            assert not foreign, f"answer for {word!r} carried {foreign}: {content[:120]!r}"

    def test_long_prompt_context_reaches_decode(self, setup_backend):
        _, model, client, gateway = setup_backend
        secret = "48213"
        computed_before = self._decode_computed(gateway) if _moriio_lane() else 0.0

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _long_prompt(secret)}],
            temperature=0,
            max_tokens=128,
            timeout=120.0,
        )

        text = _answer_text(response.choices[0].message.model_dump())
        assert secret in text, f"decode did not see the prefilled context: {text[:200]!r}"
        if _moriio_lane():
            # Decode took the prompt's KV over MoRI-IO (or, on a rerun, from its
            # own prefix cache) instead of computing it: READ leaves it the last
            # token, WRITE none. The counters can lag the response by a moment.
            allowed = 1 if vllm_moriio_mode() == "read" else 0
            deadline = time.monotonic() + 5.0
            computed = self._decode_computed(gateway) - computed_before
            while computed > allowed and time.monotonic() < deadline:
                time.sleep(0.5)
                computed = self._decode_computed(gateway) - computed_before
            assert computed <= allowed, (
                f"decode computed {computed:.0f} of the prompt's "
                f"{response.usage.prompt_tokens} tokens instead of receiving their KV"
            )

    @staticmethod
    def _decode_computed(gateway: Gateway) -> float:
        """Prompt tokens the decode workers computed themselves."""
        return sum(
            _worker_metric(w, "vllm:prompt_tokens_by_source_total", source="local_compute")
            for w in gateway.decode_workers
        )

    def test_second_turn_sees_the_first(self, setup_backend):
        _, model, client, gateway = setup_backend
        first = {"role": "user", "content": "Remember this: the password is marigold. Reply OK."}
        reply = client.chat.completions.create(
            model=model, messages=[first], temperature=0, max_tokens=64, timeout=120.0
        )
        assistant = reply.choices[0].message

        follow_up = client.chat.completions.create(
            model=model,
            messages=[
                first,
                {"role": "assistant", "content": assistant.content or "OK"},
                {"role": "user", "content": "What is the password? Reply with one word."},
            ],
            temperature=0,
            max_tokens=64,
            timeout=120.0,
        )

        text = _answer_text(follow_up.choices[0].message.model_dump())
        assert "marigold" in text.lower(), f"second turn lost the first: {text[:200]!r}"

    # -- streaming ---------------------------------------------------------

    def test_stream_completes_with_usage(self, setup_backend):
        _, model, client, gateway = setup_backend

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count from one to ten in words."}],
            temperature=0,
            max_tokens=64,
            stream=True,
            stream_options={"include_usage": True},
            timeout=120.0,
        )
        text, finish, usage = "", None, None
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            for choice in chunk.choices:
                # Thinking models may spend the whole budget in the reasoning
                # channel; either channel proves the decode side streamed.
                delta = choice.delta
                text += delta.content or ""
                text += (
                    getattr(delta, "reasoning_content", None)
                    or getattr(delta, "reasoning", None)
                    or ""
                )
                finish = choice.finish_reason or finish

        assert text.strip(), "stream carried no content"
        assert finish in ("stop", "length"), f"stream ended without a finish reason: {finish}"
        assert usage is not None and usage.completion_tokens > 0, f"no usage on the stream: {usage}"

    def test_abandoned_streams_free_both_sides(self, setup_backend):
        backend, model, client, gateway = setup_backend
        _require_load_reports(backend)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Write a very long story about the sea."}],
            "max_tokens": 1500,
            "temperature": 0,
            "stream": True,
            "ignore_eos": True,
        }

        def _abandon() -> None:
            with httpx.stream(
                "POST", f"{gateway.base_url}/v1/chat/completions", json=body, timeout=60.0
            ) as resp:
                assert resp.status_code == 200, resp.read()[:200]
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        break  # first token seen; drop the connection

        errors = _run_all([_abandon for _ in range(4)], timeout=90)
        assert not errors, f"streams failed before they could be abandoned: {errors[:2]}"

        _assert_fleet_idle_within(gateway, 30.0)
        _wait_until_served(gateway, model, timeout=60.0)

    def test_aborts_during_prefill_free_both_sides(self, setup_backend):
        """Clients that give up before the first token leave no room behind."""
        backend, model, _, gateway = setup_backend
        _require_load_reports(backend)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": _long_prompt("22222")}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
        }

        def _give_up_early() -> None:
            try:
                httpx.post(
                    f"{gateway.base_url}/v1/chat/completions",
                    json=body,
                    timeout=httpx.Timeout(10.0, read=0.3),
                )
            except httpx.ReadTimeout:
                return  # dropped while the prompt was still being prefilled
            # An answer inside the read window is not a failure of this test.

        errors = _run_all([_give_up_early for _ in range(8)], timeout=60)
        assert not errors, f"requests failed before they could be abandoned: {errors[:2]}"
        _assert_fleet_idle_within(gateway, 30.0)
        _wait_until_served(gateway, model, timeout=60.0)

    def test_batched_completion_serves_every_choice(self, setup_backend, request):
        """Every choice of an ``n>1`` request must come back through the PD pair.

        On the room-based engines (SGLang, TokenSpeed) the gRPC router fans
        the request out into one single-sample pair per choice, each with its
        own bootstrap room (#2482); vLLM dispatches sequentially and skips the
        handoff for ``n>1``, leaving decode to recompute the prompt.
        """
        mode, model, _, gateway = setup_backend
        if _moriio_lane():
            # One MoRI-IO handoff moves one request's KV, so SMG refuses n>1
            # rather than letting the samples share a prefill.
            resp = httpx.post(
                f"{gateway.base_url}/v1/completions",
                json={"model": model, "prompt": "The three primary colors are", "n": 4},
                timeout=60.0,
            )
            assert resp.status_code == 400, f"{resp.status_code} {resp.text[:300]}"
            assert _error_code(resp) == "moriio_fanout_unsupported", resp.text[:300]
            return
        if mode == "pd_http" and is_sglang():
            # The shape the gRPC fan-out fixed: the HTTP PD router mints one
            # room for a single-prompt request whatever ``n`` is, the engine
            # broadcasts it to every sample, and the decode's children wait on
            # a transfer that never comes until the decode aborts them. vLLM
            # over HTTP skips the handoff for n>1 and lets decode own the prompt.
            request.node.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    reason="n>1 over HTTP PD shares one bootstrap room across samples and hangs",
                )
            )

        resp = httpx.post(
            f"{gateway.base_url}/v1/completions",
            json={
                "model": model,
                "prompt": "The three primary colors are",
                "n": 4,
                "max_tokens": 16,
                "temperature": 0.8,
            },
            timeout=60.0,
        )

        assert resp.status_code == 200, f"{resp.status_code} {resp.text[:300]}"
        choices = resp.json().get("choices", [])
        assert len(choices) == 4, f"expected 4 choices, got {len(choices)}"
        assert all(c.get("text", "").strip() for c in choices), f"empty choice in {choices}"

    # -- failure -----------------------------------------------------------

    def _pool_survives_losing_one(self, setup_backend, role: str) -> None:
        _, model, client, gateway = setup_backend
        workers = gateway.prefill_workers if role == "prefill" else gateway.decode_workers
        if len(workers) < 2:
            pytest.skip(f"topology has a single {role} worker")
        if _moriio_holds_memory_of(role):
            pytest.skip(_MORIIO_HELD_MEMORY)
        victim = workers[-1]
        peers = {w.base_url for w in workers[:-1]}

        victim.stop()
        _wait_for_status(gateway, victim.base_url, "unhealthy", timeout=30.0)
        before = len(_pairs_logged())
        for i in range(6):
            _ask(client, model, f"Reply with the number {i}.", max_tokens=8)
        pairs = _wait_for_pairs(before + 6)[before:]
        used = {p for p, _ in pairs} if role == "prefill" else {d for _, d in pairs}
        assert victim.base_url not in used, f"gateway kept routing to the dead {role} worker"
        assert used <= peers, f"unknown {role} workers in placement: {used - peers}"

        victim.start(timeout=_RESTART_TIMEOUT)
        # start() blocked on the worker's own health; the gateway's monitor
        # notices within a few of its 1 s intervals.
        _wait_for_status(gateway, victim.base_url, "healthy", timeout=60.0)
        before = len(_pairs_logged())
        count = 4 * len(workers)
        for i in range(count):
            _ask(client, model, f"Reply with the number {i}.", max_tokens=8)
        pairs = _wait_for_pairs(before + count)[before:]
        used = {p for p, _ in pairs} if role == "prefill" else {d for _, d in pairs}
        assert victim.base_url in used, f"restarted {role} worker never rejoined rotation"

    @pytest.mark.flaky(reruns=0)  # a restart per attempt; a real failure must surface once
    def test_decode_pool_survives_losing_one_worker(self, setup_backend):
        self._pool_survives_losing_one(setup_backend, "decode")

    @pytest.mark.flaky(reruns=0)  # a restart per attempt; a real failure must surface once
    def test_prefill_pool_survives_losing_one_worker(self, setup_backend):
        self._pool_survives_losing_one(setup_backend, "prefill")

    @pytest.mark.flaky(reruns=0)  # a restart per attempt; a real failure must surface once
    def test_decode_engine_death_under_streaming_traffic(self, setup_backend):
        """A decode engine dies under live streams.

        Each stream ends complete or with the gateway's error frame and
        ``[DONE]``, never silently and never as a completion the engine
        aborted; the gateway stops routing to the worker and its peers answer
        new requests; the worker's restart on the same GPU rejoins rotation.

        The fault is the engine's own processes dying under their launcher,
        what a device fault or the kernel's out-of-memory killer does, not an
        orderly stop: the launcher reports the death and exits, and the
        harness restarts the worker once its ranks have released the device
        memory. Over a KV transport whose peers keep a dead engine's memory
        mapped, the restart fails its startup memory check here: that is
        this row's finding about the transport, not a reason to skip.
        """
        mode, model, client, gateway = setup_backend
        if mode != "pd_grpc":
            # The MoRI-IO lanes run on the HTTP PD router and skip here too.
            pytest.skip(
                "the stream-failure contract asserted here is the gRPC pipeline's; "
                "the HTTP PD router relays the engine's own frames"
            )
        decodes = gateway.decode_workers
        if len(decodes) < 2:
            pytest.skip("topology has a single decode worker")
        victim = decodes[-1]
        if is_sglang() and victim.effective_kv_backend() == "nixl":
            # The nightly SGLang-over-NIXL row keeps the other peer restarts
            # out by name for the same reason.
            pytest.skip(
                "the SGLang prefill segfaults when its decode comes back on the same "
                "address (#2503), and this row restarts the decode on its old port"
            )
        peers = {w.base_url for w in decodes[:-1]}

        streams = 4 * len(decodes)
        before = len(_pairs_logged())
        outcomes: list[_StreamOutcome] = []
        first_tokens = [threading.Event() for _ in range(streams)]

        def _run(event: threading.Event) -> None:
            outcomes.append(_stream_chat(gateway, model, max_tokens=600, first_token=event))

        threads = [threading.Thread(target=_run, args=(e,), daemon=True) for e in first_tokens]
        for t in threads:
            t.start()
        # Every stream is past its first token before the kill, so the engine
        # dies under live decode, on the victim and on its peers alike.
        deadline = time.monotonic() + 60.0
        for event in first_tokens:
            assert event.wait(max(0.0, deadline - time.monotonic())), (
                "a stream had no first token within 60s"
            )
        pairs = _wait_for_pairs(before + streams)[before:]
        decodes_used = {d for _, d in pairs}
        assert victim.base_url in decodes_used, f"no stream landed on {victim.base_url}: {pairs}"

        killed = victim.kill_engine()
        assert killed, f"{victim.base_url}: the launcher had no engine process to kill"
        logger.info(
            "%s: engine processes %s killed under %d streams", victim.base_url, killed, streams
        )

        for t in threads:
            t.join(timeout=120.0)
        try:
            assert len(outcomes) == streams, (
                f"{streams - len(outcomes)} streams still running 120s after the kill"
            )
            logger.info("stream outcomes after the kill: %s", outcomes)
            exceptions = [o for o in outcomes if o.exception is not None]
            assert not exceptions, f"streams failed below the SSE layer: {exceptions}"
            ends = [o.end for o in outcomes]
            aborted = [e for e in ends if e.finish_reason == "abort"]
            assert not aborted, f"streams ended as completions the engine aborted: {aborted}"
            silent = [
                e for e in ends if e.error is None and e.finish_reason not in ("stop", "length")
            ]
            assert not silent, (
                f"streams ended with neither a completion nor an error frame: {silent}"
            )
            failed = [e for e in ends if e.error is not None]
            assert failed, "no stream reported the engine's death"
            undone = [e for e in failed if not e.done]
            assert not undone, f"failed streams not closed with [DONE]: {undone}"

            # The gateway takes the worker out of rotation on its own and the
            # peers answer: no client reads an error for a worker that is gone.
            _wait_for_status(gateway, victim.base_url, "unhealthy", timeout=30.0)
            before = len(_pairs_logged())
            for i in range(6):
                _ask(client, model, f"Reply with the number {i}.", max_tokens=8)
            pairs = _wait_for_pairs(before + 6)[before:]
            used = {d for _, d in pairs}
            assert victim.base_url not in used, "gateway kept routing to the dead decode worker"
            assert used <= peers, f"unknown decode workers in placement: {used - peers}"
        finally:
            # Whatever the verdict, the class's later tests need the pool back.
            # stop() reaps the engine's processes and waits for the device
            # memory before the same GPU is loaded again.
            victim.stop()
            victim.start(timeout=_RESTART_TIMEOUT)
            _wait_for_status(gateway, victim.base_url, "healthy", timeout=60.0)
        before = len(_pairs_logged())
        count = 4 * len(decodes)
        for i in range(count):
            _ask(client, model, f"Reply with the number {i}.", max_tokens=8)
        pairs = _wait_for_pairs(before + count)[before:]
        assert victim.base_url in {d for _, d in pairs}, (
            "restarted decode worker never rejoined rotation"
        )

    # The name keeps "leg": two workflows select this test by `not sole_leg_outage`
    # (the nightly PD sweep and the PR e2e matrix); a rename moves those filters with it.
    @pytest.mark.flaky(reruns=0)  # a restart per attempt; a real failure must surface once
    def test_sole_leg_outage_is_reported_promptly(self, setup_backend):
        _, model, client, gateway = setup_backend
        if len(gateway.decode_workers) == 1:
            role, victim = "decode", gateway.decode_workers[0]
        elif len(gateway.prefill_workers) == 1:
            role, victim = "prefill", gateway.prefill_workers[0]
        else:
            pytest.skip("both pools have more than one worker")
        if _moriio_holds_memory_of(role):
            pytest.skip(_MORIIO_HELD_MEMORY)
        if _nixl_decodes_keep_a_dead_prefills_memory(role):
            pytest.skip(_NIXL_HELD_MEMORY)

        victim.stop()
        try:
            _wait_for_status(gateway, victim.base_url, "unhealthy", timeout=30.0)
            started = time.monotonic()
            resp = _raw_chat(gateway, model, "Say hello.", timeout=60.0)
            elapsed = time.monotonic() - started
            code = _error_code(resp)
            logger.info(
                "sole %s down: status=%s code=%s after %.1fs", role, resp.status_code, code, elapsed
            )
            assert elapsed < 20.0, f"request hung for {elapsed:.1f}s while the only {role} was down"
            # The model exists and its pool is merely down: a 503 the client can
            # retry, never a 404 that says the model is gone (#2465). Every
            # router answers no_available_workers (#2479); the HTTP PD router
            # may name the role instead.
            assert resp.status_code == 503, (
                f"unexpected outage answer: {resp.status_code} {resp.text[:200]}"
            )
            assert code in (
                "no_available_workers",
                f"no_{role}_servers",
                f"{role}_unavailable",
            ), f"outage answer carried an unexpected code: {code!r} {resp.text[:200]}"
        finally:
            # Whatever the verdict, the class's later tests need the pool back.
            victim.start(timeout=_RESTART_TIMEOUT)
            _wait_for_status(gateway, victim.base_url, "healthy", timeout=60.0)
        _wait_until_served(gateway, model, timeout=120.0)


# ---------------------------------------------------------------------------
# a fleet that mixes KV transfer backends
# ---------------------------------------------------------------------------


@pytest.mark.engine("vllm")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.workers(parallel_start=True)
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR), extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize(
    "setup_backend",
    [
        pytest.param(
            (
                "pd_grpc",
                (2, 2, {"prefill_kv": ["nixl", "mooncake"], "decode_kv": ["nixl", "mooncake"]}),
            ),
            id="2p2d-mixed-kv",
        )
    ],
    indirect=True,
)
@_NEEDS_BOTH_VLLM_TRANSPORTS
class TestPDMixedTransport:
    """One NIXL pair and one Mooncake pair share a model.

    Placement pairs a prefill with a decode on their KV transfer protocol
    (#2483): a NIXL prefill's handoff never lands on a Mooncake decode, and
    the gateway reports which protocol each worker speaks.
    """

    def test_every_request_lands_on_a_matching_pair(self, setup_backend):
        _, model, _, gateway = setup_backend
        _wait_for_healthy_workers(gateway, 4, timeout=120.0)
        keys = _pairing_keys(gateway)
        logger.info("mixed fleet pairing keys: %s", keys)
        assert len(keys) == 4, keys
        transports = {key.split("/")[1] for key in keys.values()}
        assert transports == {"nixl", "mooncake"}, keys

        before = len(_pairs_logged())
        statuses = []
        for i in range(12):
            resp = _raw_chat(gateway, model, f"Say hello, {_WORDS[i % len(_WORDS)]}.", timeout=60.0)
            statuses.append((resp.status_code, _error_code(resp)))
        pairs = _wait_for_pairs(before + 12)[before:]
        logger.info("mixed fleet: statuses=%s pairs=%s", statuses, pairs)
        failed = [s for s in statuses if s[0] != 200]
        assert not failed, f"{len(failed)} of 12 requests failed on a mixed fleet: {failed}"
        crossed = [(p, d) for p, d in pairs if keys.get(p) != keys.get(d)]
        assert not crossed, f"pairs crossed KV transports: {crossed}"
        logger.info(
            "mixed fleet transports served: %s",
            sorted({keys[p].split("/")[1] for p, _ in pairs if p in keys}),
        )


@pytest.mark.engine("vllm")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.workers(parallel_start=True)
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR), extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize(
    "setup_backend",
    [
        pytest.param(
            ("pd_grpc", (1, 1, {"prefill_kv": ["nixl"], "decode_kv": ["mooncake"]})),
            id="1p1d-kv-mismatch",
        )
    ],
    indirect=True,
)
@_NEEDS_BOTH_VLLM_TRANSPORTS
class TestPDMismatchedTransport:
    """A NIXL prefill and a Mooncake decode can never complete a handoff.

    Placement refuses the pair up front with `no_compatible_pd_pair`
    (#2483) instead of letting the engine time out on the rendezvous.
    """

    def test_request_is_refused_at_placement(self, setup_backend):
        _, model, _, gateway = setup_backend
        _wait_for_healthy_workers(gateway, 2, timeout=120.0)
        keys = _pairing_keys(gateway)
        logger.info("mismatched fleet pairing keys: %s", keys)
        assert len(keys) == 2, keys

        resp = _raw_chat(gateway, model, "Say hello.", timeout=60.0)
        assert resp.status_code == 503, resp.text
        assert _error_code(resp) == "no_compatible_pd_pair", resp.text


# ---------------------------------------------------------------------------
# an explicit pairing protocol injected into the engines
# ---------------------------------------------------------------------------


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model(_MODEL, tokenspeed=_MODEL_BY_ENGINE["tokenspeed"])
@pytest.mark.workers(parallel_start=True)
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR), extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize(
    "setup_backend",
    [
        pytest.param(
            (
                "pd_grpc",
                (
                    1,
                    1,
                    {
                        "prefill_env": {"SMG_PAIRING_PROTOCOL": "kv-v1"},
                        "decode_env": {"SMG_PAIRING_PROTOCOL": "kv-v2"},
                    },
                ),
            ),
            id="1p1d-protocol-mismatch",
        )
    ],
    indirect=True,
)
@pytest.mark.skipif(
    _moriio_lane(),
    reason="the gRPC servicer reports SMG_PAIRING_PROTOCOL; a MoRI-IO lane's workers are HTTP",
)
class TestPDMismatchedProtocol:
    """A pairing protocol set in the engine's environment reaches placement.

    A deployment injects ``SMG_PAIRING_PROTOCOL`` into the engine container;
    the servicer reports it in server info and the gateway pairs on it alone.
    Two workers with different values are refused with `no_compatible_pd_pair`,
    whatever their transport and layout say.
    """

    def test_workers_with_different_protocols_are_refused(self, setup_backend):
        _, model, _, gateway = setup_backend
        _wait_for_healthy_workers(gateway, 2, timeout=120.0)
        keys = _pairing_keys(gateway)
        logger.info("explicit-protocol fleet pairing keys: %s", keys)
        by_url = {w.url: role for role, ws in _workers_by_role(gateway).items() for w in ws}
        reported = {by_url.get(url, url): key for url, key in keys.items()}
        # The explicit value is the whole key: the engine's environment, as
        # the servicer reported it, replaces runtime/transport/layout.
        assert reported == {"prefill": "kv-v1", "decode": "kv-v2"}, keys

        resp = _raw_chat(gateway, model, "Say hello.", timeout=60.0)
        assert resp.status_code == 503, resp.text
        assert _error_code(resp) == "no_compatible_pd_pair", resp.text


# ---------------------------------------------------------------------------
# a decode window smaller than the burst
# ---------------------------------------------------------------------------


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model(_MODEL, tokenspeed=_MODEL_BY_ENGINE["tokenspeed"])
@pytest.mark.workers(parallel_start=True, extra_engine_args=_WINDOW_ARGS)
@pytest.mark.gateway(
    log_level="debug",
    log_dir=str(_LOG_DIR),
    extra_args=[*_GATEWAY_ARGS, "--pd-admission-wait-secs", "2"],
)
@pytest.mark.parametrize(
    "setup_backend", [pytest.param(("pd_grpc", (1, 1)), id="1p1d-window4")], indirect=True
)
class TestPDSmallWindow:
    """A burst wider than the decode window is held or shed at the gateway.

    Without the admission gate the excess reached the engine, where the
    prefill's bootstrap deadline expired on requests that were merely queued
    behind the decode's admission, and every such room then held a decode slot
    for the whole transfer timeout (run 34173426995: 14 of 64 requests lost,
    1392 s for an eval that takes 35 s).
    """

    def test_burst_beyond_decode_window_is_shed_not_stalled(self, setup_backend):
        _, model, _, gateway = setup_backend
        _wait_until_served(gateway, model, timeout=60.0)
        # The window must have reached the engine, or the burst fits and the
        # gate is never crossed. Engines that report their window on /loads
        # (SGLang, TokenSpeed) are checked; vLLM reports none. Every worker has
        # to appear in /loads first, which takes the monitor an interval.
        deadline = time.monotonic() + 30.0
        reported, loads = _fleet_idle(gateway)
        while not reported and time.monotonic() < deadline:
            time.sleep(1.0)
            reported, loads = _fleet_idle(gateway)
        assert reported, f"a worker never reported a load: {loads}"
        decode_urls = {w.base_url for w in gateway.decode_workers}
        windows = {
            e["worker"]: e["max_running_requests"]
            for e in loads
            if e.get("worker") in decode_urls and e.get("max_running_requests", 0) > 0
        }
        assert all(w == _WINDOW for w in windows.values()), f"decode window not applied: {windows}"
        logger.info("decode windows reported: %s", windows or "none (engine reports no window)")
        worker_dir = worker_log_dir(_LOG_DIR)
        before = _log_sizes(worker_dir)
        results: list[httpx.Response] = []

        def _one(i: int) -> None:
            results.append(
                httpx.post(
                    f"{gateway.base_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "user", "content": f"Write {i} sentences about the sea."}
                        ],
                        "max_tokens": 96,
                        "temperature": 0,
                        "ignore_eos": True,
                    },
                    timeout=120.0,
                )
            )

        started = time.monotonic()
        errors = _run_all([partial(_one, i) for i in range(6 * _WINDOW)], timeout=180)
        elapsed = time.monotonic() - started

        assert not errors, f"requests failed at the transport: {errors[:2]}"
        statuses = Counter(r.status_code for r in results)
        logger.info(
            "burst of %d against a window of %d: %s in %.1fs",
            6 * _WINDOW,
            _WINDOW,
            dict(statuses),
            elapsed,
        )
        assert set(statuses) <= {200, 503}, (
            f"a burst must be served or shed, never errored: {dict(statuses)}"
        )
        assert statuses[200] >= _WINDOW, f"too few requests served: {dict(statuses)}"
        for resp in results:
            if resp.status_code == 503:
                assert _error_code(resp) == "worker_overload_protection_shed", resp.text[:200]
                assert resp.headers.get("retry-after"), "a shed must say when to retry"
        assert elapsed < 90.0, f"the burst took {elapsed:.0f}s; queued rooms are timing out"

        # Only what this burst appended, per file: earlier classes kill workers
        # out from under their peers. The capture guard checks this fleet's own
        # files exist, so an uncaptured worker cannot pass as a quiet one.
        assert_worker_logs_captured(_fleet_logs(before, gateway), "bootstrap timeouts")
        new_lines = _logs_since(worker_dir, before)
        for marker in _BOOTSTRAP_TIMEOUT_MARKERS:
            assert marker not in new_lines, f"an engine worker timed out a bootstrap: {marker!r}"
        _wait_until_served(gateway, model, timeout=60.0)


# ---------------------------------------------------------------------------
# operations: a PD fleet assembled at runtime
# ---------------------------------------------------------------------------


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model(_MODEL, tokenspeed=_MODEL_BY_ENGINE["tokenspeed"])
class TestPDAssembledAtRuntime:
    """IGW gateway with no workers; prefill and decode arrive through the API."""

    @pytest.mark.flaky(reruns=0)  # a restart per attempt; a real failure must surface once
    def test_workers_added_through_the_api_route_as_pd(self):
        engine = get_runtime()
        model_id = _MODEL_BY_ENGINE.get(engine, _MODEL)
        model_path = get_model_spec(model_id)["model"]
        cleanup_pool()  # the sweep above owns no cached workers, but be explicit about the GPUs
        log_dir = os.environ.get("E2E_LOG_DIR")
        moriio = _moriio_lane()
        # MoRI-IO workers are HTTP workers registered with their handoff fields.
        mode = ConnectionMode.HTTP if moriio else ConnectionMode.GRPC
        prefill: list = []
        decode: list = []
        gateway = Gateway()
        try:
            prefill = start_workers(
                model_id,
                engine,
                mode=mode,
                count=1,
                worker_type=WorkerType.PREFILL,
                log_dir=log_dir,
            )
            decode = start_workers(
                model_id,
                engine,
                mode=mode,
                count=1,
                worker_type=WorkerType.DECODE,
                log_dir=log_dir,
                gpu_offset=1,
            )
            gateway.start(
                igw_mode=True, log_level="debug", log_dir=str(_LOG_DIR), extra_args=_HEALTH_ARGS
            )
            before = len(_pairs_logged())

            def _register(worker, worker_type: str) -> tuple[bool, str | None]:
                if moriio:
                    spec = worker.moriio_registration()
                    return gateway.add_worker(worker.base_url, worker_type=worker_type, spec=spec)
                return gateway.add_worker(
                    worker.base_url, worker_type=worker_type, bootstrap_port=worker.bootstrap_port
                )

            ok, detail = _register(prefill[0], "prefill")
            assert ok, f"registering the prefill worker failed: {detail}"
            ok, detail = _register(decode[0], "decode")
            assert ok, f"registering the decode worker failed: {detail}"
            roles = {r: len(ws) for r, ws in _workers_by_role(gateway).items()}
            logger.info("roles after registration: %s", roles)
            assert roles == {"prefill": 1, "decode": 1}, f"/workers roles: {roles}"

            _wait_until_served(gateway, model_path, timeout=180.0)
            pairs = _wait_for_pairs(before + 1)[before:]
            assert (prefill[0].base_url, decode[0].base_url) in pairs, (
                f"request did not go through the registered pair; pairs={pairs}"
            )

            ok, detail = gateway.remove_worker(decode[0].base_url)
            assert ok, f"removing the decode worker failed: {detail}"
            # The removal is accepted (202) and drained in the background; the
            # outage exists once the worker has left /workers.
            _wait_for_removal(gateway, decode[0].base_url, timeout=60.0)
            started = time.monotonic()
            resp = _raw_chat(gateway, model_path, "Say hello.", timeout=60.0)
            elapsed = time.monotonic() - started
            logger.info(
                "decode removed: status=%s code=%s after %.1fs",
                resp.status_code,
                _error_code(resp),
                elapsed,
            )
            assert resp.status_code != 200 and elapsed < 20.0, (
                f"with no decode worker the gateway answered {resp.status_code} after {elapsed:.1f}s"
            )

            ok, detail = _register(decode[0], "decode")
            assert ok, f"re-registering the decode worker failed: {detail}"
            _wait_until_served(gateway, model_path, timeout=120.0)
        finally:
            gateway.shutdown()
            stop_workers(prefill + decode)
            logger.info("router logs kept at %s", _LOG_DIR)
