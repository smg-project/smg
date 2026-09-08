"""Prefill/decode disaggregation under every worker topology a 4-GPU node allows.

One class sweeps the topologies 1p1d, 1p2d, 2p1d, 1p3d, 3p1d and 2p2d. The
counts ride in the ``setup_backend`` param because the fixture is
class-scoped, and every test runs once per topology. The gateway logs each
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

import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from infra import ConnectionMode, Gateway, WorkerType, cleanup_pool, start_workers, stop_workers
from infra.constants import get_runtime
from infra.model_specs import get_model_spec
from infra.pd_logs import LOG_FLUSH_TIMEOUT_S, read_logs

logger = logging.getLogger(__name__)

# Router logs land here via the gateway marker (rolling files named smg.YYYY-MM-DD)
_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-pd-topologies-{os.getpid()}"
PAIR_MARKER = "Selected PD pair"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PAIR_RE = re.compile(r"prefill=(\S+)\s+decode=(\S+)")

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_MODEL_BY_ENGINE = {"tokenspeed": "Qwen/Qwen3.5-9B"}
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
_TOPOLOGIES = [
    pytest.param(("pd_grpc", (p, d)), id=f"{p}p{d}d")
    for p, d in [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
]
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
    busy = [e for e in loads if e.get("num_running_reqs", 0) + e.get("num_waiting_reqs", 0) > 0]
    return (not busy, loads)


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
    for t in threads:
        t.join(timeout=timeout)
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
        _, model, _, gateway = setup_backend
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
                text += choice.delta.content or ""
                finish = choice.finish_reason or finish

        assert text.strip(), "stream carried no content"
        assert finish in ("stop", "length"), f"stream ended without a finish reason: {finish}"
        assert usage is not None and usage.completion_tokens > 0, f"no usage on the stream: {usage}"

    def test_abandoned_streams_free_both_legs(self, setup_backend):
        _, model, client, gateway = setup_backend
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

        deadline = time.monotonic() + 30.0
        idle, loads = _fleet_idle(gateway)
        while not idle and time.monotonic() < deadline:
            time.sleep(1.0)
            idle, loads = _fleet_idle(gateway)
        assert idle, f"abandoned streams still occupy the engines after 30s: {loads}"
        _wait_until_served(gateway, model, timeout=60.0)

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

        victim.start()
        _wait_for_status(gateway, victim.base_url, "healthy", timeout=300.0)
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
        _wait_for_status(gateway, victim.base_url, "unhealthy", timeout=30.0)
        started = time.monotonic()
        resp = _raw_chat(gateway, model, "Say hello.", timeout=60.0)
        elapsed = time.monotonic() - started
        code = _error_code(resp)
        logger.info(
            "sole %s down: status=%s code=%s after %.1fs", role, resp.status_code, code, elapsed
        )
        assert elapsed < 20.0, f"request hung for {elapsed:.1f}s while the only {role} was down"
        # The gRPC path answers 404 model_not_found for an unavailable leg today;
        # the HTTP router answers 503 no_available_workers. Either is prompt and
        # carries a code; a follow-up aligns the gRPC path with the 503.
        assert resp.status_code in (404, 503), (
            f"unexpected outage answer: {resp.status_code} {resp.text[:200]}"
        )
        assert code, f"outage answer carried no error code: {resp.text[:200]}"

        victim.start()
        _wait_for_status(gateway, victim.base_url, "healthy", timeout=300.0)
        _wait_until_served(gateway, model, timeout=120.0)


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
        gateway = Gateway()
        try:
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
