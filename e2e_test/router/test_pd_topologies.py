"""Prefill/decode disaggregation under every worker topology a 4-GPU node allows.

One class sweeps the topologies 1p1d, 1p2d, 2p1d, 1p3d, 3p1d and 2p2d, two
asymmetric-tp pairs (1p2d-ptp2-dtp1, 2p1d-ptp1-dtp2) and the HTTP PD router's
1p1d and 2p2d. The counts, and the per-leg tp when the legs differ, ride in
the ``setup_backend`` param because the fixture is class-scoped, and every
test runs once per topology. The gateway logs each
selected pair at debug level, so requests are attributed to workers from the
router log instead of guessed.

What a user of PD mode hits, in the order they hit it:

- registration: ``/workers`` lists every leg with its role, all healthy
- placement: every prefill and every decode worker takes traffic
- transfer: a long prompt's context reaches decode; concurrent requests do
  not see each other's KV; a second turn still sees the first
- streaming: a stream completes with usage; an abandoned stream frees the
  engine slots on both legs instead of generating to its limit
- failure: a leg loses one worker and traffic keeps flowing; a leg loses its
  only worker and the client gets a prompt error instead of a hang; the
  worker comes back and rejoins rotation
- operations: a PD fleet assembled at runtime through the worker API in IGW
  mode routes through prefill and decode

TokenSpeed runs Qwen3.5-9B, the model its disaggregation path is proven on;
SGLang and vLLM run Llama-3.1-8B.

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
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from infra import ConnectionMode, Gateway, WorkerType, cleanup_pool, start_workers, stop_workers
from infra.constants import DEFAULT_STARTUP_TIMEOUT, get_runtime, is_sglang
from infra.model_specs import get_model_spec
from infra.pd_logs import (
    LOG_FLUSH_TIMEOUT_S,
    assert_worker_logs_captured,
    read_logs,
    worker_log_dir,
)

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
        pytest.param(("pd_grpc", (p, d, ptp, dtp)), id=f"{p}p{d}d-ptp{ptp}-dtp{dtp}")
        for p, d, ptp, dtp in [(1, 2, 2, 1), (2, 1, 1, 2)]
    ]
    + [
        # The HTTP PD router is its own code path: pairing and error answers
        # on both engines, the KV handoff on SGLang. A vLLM leg registered by
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
            # A restart truncates the log (mode "w"): a stale offset past EOF
            # would silently read nothing, so fall back to the whole file.
            start = sizes.get(path, 0)
            fh.seek(start if start <= path.stat().st_size else 0)
            parts.append(fh.read().decode("utf-8", errors="replace"))
    return "\n".join(parts)


def _fleet_logs(sizes: dict[Path, int], gateway: Gateway) -> str:
    """The captured log files of this gateway's own legs, named by port."""
    legs = gateway.prefill_workers + gateway.decode_workers
    ports = {str(urlparse(w.base_url).port) for w in legs}
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


def _fleet_idle(gateway: Gateway) -> tuple[bool, list[dict]]:
    resp = httpx.get(f"{gateway.base_url}/loads", timeout=5.0)
    assert resp.status_code == 200, resp.text
    loads = resp.json().get("loads", [])
    # /loads omits workers the monitor has no report for; an unreported leg
    # is not evidence of idleness.
    reported = {e.get("worker") for e in loads}
    expected = {w.base_url for w in gateway.prefill_workers + gateway.decode_workers}
    if not expected <= reported:
        return (False, loads)
    busy = [e for e in loads if e.get("num_running_reqs", 0) + e.get("num_waiting_reqs", 0) > 0]
    return (not busy, loads)


def _require_load_reports(backend: str) -> None:
    """The idle check reads /loads; HTTP SGLang workers report none without metrics."""
    if backend == "pd_http":
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
        assert not unhealthy, f"unhealthy legs after startup: {unhealthy}"
        assert any(m.get("id") for m in gateway.list_models()), "no model listed"

        # Every leg reports the KV transport the lane launched it with: a row
        # that names a transport its workers never took sweeps the wrong thing
        # (#2498). gRPC discovery reads the engine's own server args, so the
        # transport is known there; HTTP discovery's curated server_info does
        # not carry it and reads as "?". TokenSpeed is not pinned by this: it
        # only moves KV over Mooncake, so both sides are that constant.
        keys = _pairing_keys(gateway)
        logger.info("pairing keys: %s", keys)
        legs = gateway.prefill_workers + gateway.decode_workers
        reported = {w.base_url: keys.get(w.base_url, "?/?").split("/")[1] for w in legs}
        launched = {w.base_url: w.effective_kv_backend() for w in legs}
        wrong = {
            u: (reported[u], launched[u]) for u in launched if reported[u] not in ("?", launched[u])
        }
        assert not wrong, (
            f"legs report a different KV transport than they were launched with: {wrong}"
        )
        if mode == "pd_grpc":
            unknown = [u for u, t in reported.items() if t == "?"]
            assert not unknown, f"gRPC legs without a known KV transport on /workers: {unknown}"

    # -- placement ---------------------------------------------------------

    def test_every_leg_takes_traffic(self, setup_backend):
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

        message = _ask(client, model, _long_prompt(secret), max_tokens=128)

        text = _answer_text(message)
        assert secret in text, f"decode did not see the prefilled context: {text[:200]!r}"

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
                # channel; either channel proves the decode leg streamed.
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

    def test_abandoned_streams_free_both_legs(self, setup_backend):
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

    def test_aborts_during_prefill_free_both_legs(self, setup_backend):
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

    def _leg_survives_losing_one(self, setup_backend, role: str) -> None:
        _, model, client, gateway = setup_backend
        workers = gateway.prefill_workers if role == "prefill" else gateway.decode_workers
        if len(workers) < 2:
            pytest.skip(f"topology has a single {role} worker")
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
    def test_decode_leg_survives_losing_one_worker(self, setup_backend):
        self._leg_survives_losing_one(setup_backend, "decode")

    @pytest.mark.flaky(reruns=0)  # a restart per attempt; a real failure must surface once
    def test_prefill_leg_survives_losing_one_worker(self, setup_backend):
        self._leg_survives_losing_one(setup_backend, "prefill")

    @pytest.mark.flaky(reruns=0)  # a restart per attempt; a real failure must surface once
    def test_sole_leg_outage_is_reported_promptly(self, setup_backend):
        _, model, client, gateway = setup_backend
        if len(gateway.decode_workers) == 1:
            role, victim = "decode", gateway.decode_workers[0]
        elif len(gateway.prefill_workers) == 1:
            role, victim = "prefill", gateway.prefill_workers[0]
        else:
            pytest.skip("both legs have more than one worker")

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
            # The model exists and its leg is merely down: a 503 the client can
            # retry, never a 404 that says the model is gone (#2465). Every
            # router answers no_available_workers (#2479); the HTTP PD router
            # may name the leg instead.
            assert resp.status_code == 503, (
                f"unexpected outage answer: {resp.status_code} {resp.text[:200]}"
            )
            assert code in (
                "no_available_workers",
                f"no_{role}_servers",
                f"{role}_unavailable",
            ), f"outage answer carried an unexpected code: {code!r} {resp.text[:200]}"
        finally:
            # Whatever the verdict, the class's later tests need the leg back.
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
    the gateway reports which protocol each leg speaks.
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
        # (SGLang, TokenSpeed) are checked; vLLM reports none. Every leg has
        # to appear in /loads first, which takes the monitor an interval.
        deadline = time.monotonic() + 30.0
        reported, loads = _fleet_idle(gateway)
        while not reported and time.monotonic() < deadline:
            time.sleep(1.0)
            reported, loads = _fleet_idle(gateway)
        assert reported, f"a leg never reported a load: {loads}"
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
        # files exist, so an uncaptured leg cannot pass as a quiet one.
        assert_worker_logs_captured(_fleet_logs(before, gateway), "bootstrap timeouts")
        new_lines = _logs_since(worker_dir, before)
        for marker in _BOOTSTRAP_TIMEOUT_MARKERS:
            assert marker not in new_lines, f"an engine leg timed out a bootstrap: {marker!r}"
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
    def test_legs_added_through_the_api_route_as_pd(self):
        engine = get_runtime()
        model_id = _MODEL_BY_ENGINE.get(engine, _MODEL)
        model_path = get_model_spec(model_id)["model"]
        cleanup_pool()  # the sweep above owns no cached workers, but be explicit about the GPUs
        log_dir = os.environ.get("E2E_LOG_DIR")
        prefill: list = []
        decode: list = []
        gateway = Gateway()
        try:
            prefill = start_workers(
                model_id,
                engine,
                mode=ConnectionMode.GRPC,
                count=1,
                worker_type=WorkerType.PREFILL,
                log_dir=log_dir,
            )
            decode = start_workers(
                model_id,
                engine,
                mode=ConnectionMode.GRPC,
                count=1,
                worker_type=WorkerType.DECODE,
                log_dir=log_dir,
                gpu_offset=1,
            )
            gateway.start(
                igw_mode=True, log_level="debug", log_dir=str(_LOG_DIR), extra_args=_HEALTH_ARGS
            )
            before = len(_pairs_logged())

            ok, detail = gateway.add_worker(
                prefill[0].base_url, worker_type="prefill", bootstrap_port=prefill[0].bootstrap_port
            )
            assert ok, f"registering the prefill worker failed: {detail}"
            ok, detail = gateway.add_worker(decode[0].base_url, worker_type="decode")
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

            ok, detail = gateway.add_worker(decode[0].base_url, worker_type="decode")
            assert ok, f"re-registering the decode worker failed: {detail}"
            _wait_until_served(gateway, model_path, timeout=120.0)
        finally:
            gateway.shutdown()
            stop_workers(prefill + decode)
            logger.info("router logs kept at %s", _LOG_DIR)
