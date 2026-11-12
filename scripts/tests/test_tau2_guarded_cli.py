"""Actual pinned SDK calls with fake HTTP only; real sockets are forbidden."""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tau2"))
from datetime import UTC

import guarded_cli as gc


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("LITELLM_TELEMETRY", "False")
    for name in ("ANTHROPIC_API_KEY", "AZURE_API_KEY", "GOOGLE_API_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("SDK test attempted a real socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def configured(tmp_path):
    contexts = {
        "schema_version": 1,
        "models": {
            model: {
                "context_tokens": 10,
                "response_models": [model],
                "source_url": "https://developers.openai.com/offline-fixture",
            }
            for model in gc.MODELS.values()
        },
    }
    context_path = tmp_path / "contexts.json"
    context_path.write_text(json.dumps(contexts))
    digest = hashlib.sha256(context_path.read_bytes()).hexdigest()
    data = {
        "scope_id": "offline-two-model-scope",
        "cap_nusd": 100_000,
        "calls": 10,
        "calls_per_arm": 10,
        "models": {
            "openai/" + model: {
                "input_per_million": str(rate),
                "output_per_million": str(rate),
                "input_limit": 10,
                "output_limit": 2,
                "service_tier": "default",
            }
            for model, rate in zip(gc.MODELS.values(), [1, 2])
        },
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(data))
    policy, _ = gc.load_policy(policy_path, context_path, digest)
    ledger = gc.Ledger(tmp_path / "shared.db", policy)
    return policy_path, context_path, digest, ledger, contexts


def response(request, content="hello", usage=None, status=200):
    import httpx

    data = json.loads(request.content)
    return httpx.Response(
        status,
        json={
            "id": "offline-response",
            "object": "chat.completion",
            "created": 1,
            "model": data["model"],
            "service_tier": "default",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage
            if usage is not None
            else {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.fixture
def installed(configured):
    import httpx

    _, _, _, ledger, contexts = configured
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        return response(request)

    guard = gc.Guard(
        ledger,
        contexts,
        arm="vllm",
        domain="retail",
        invocation_id="offline-invocation",
        local_model="local-model",
        local_url="http://127.0.0.1:8000",
        transport_factory=lambda: httpx.MockTransport(handle),
    )
    try:
        guard.install()
        yield guard, sent
    finally:
        guard.close()


def user_turn(**kwargs):
    from tau2.data_model.message import AssistantMessage, SystemMessage
    from tau2.user import user_simulator

    state = types.SimpleNamespace(
        messages=[],
        system_messages=[SystemMessage(role="system", content="offline")],
        flip_roles=lambda: [],
    )
    user = types.SimpleNamespace(llm="gpt-5.2", tools=[], llm_args={"temperature": 0.0, **kwargs})
    return user_simulator.UserSimulator._generate_next_message(
        user, AssistantMessage(role="assistant", content="offline"), state
    )


def test_actual_user_method_and_two_sdk_chain(installed):
    guard, sent = installed
    token = gc.TASK.set(("task0", 0))
    try:
        assert user_turn().content == "hello"
    finally:
        gc.TASK.reset(token)
    assert len(sent) == 1 and sent[0]["max_completion_tokens"] == 2
    record = guard.ledger.audit()["requests"][0]
    assert record["model"] == "openai/gpt-5.2"
    assert record["metadata"]["role"] == "user" and record["metadata"]["task_id"] == "task0"
    assert record["state"] == "complete"
    assert record["allocated_nusd"] == 12_000  # whole verified context, not prompt usage1


def judge_turn():
    from tau2.data_model.message import UserMessage
    from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator

    return NLAssertionsEvaluator.evaluate_nl_assertions(
        [UserMessage(role="user", content="offline")], ["offline assertion"]
    )


def test_actual_required_nl_and_user_share_money_not_separate_caps(installed):
    import httpx

    guard, sent = installed

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        content = (
            "hello"
            if body["model"] == "gpt-5.2"
            else json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "offline assertion",
                            "metExpectation": True,
                            "reasoning": "offline",
                        }
                    ]
                }
            )
        )
        return response(request, content)

    guard.transport_factory = lambda: httpx.MockTransport(handle)
    token = gc.TASK.set(("task0", 1))
    try:
        user_turn()
        assert judge_turn()[0].met is True
    finally:
        gc.TASK.reset(token)
    audit = guard.ledger.audit()
    assert {r["metadata"]["role"] for r in audit["requests"]} == {"user", "nl_judge"}
    assert {r["model"] for r in audit["requests"]} == {
        "openai/gpt-5.2",
        "openai/gpt-4.1-2025-04-14",
    }
    assert guard.ledger.summary()["allocated_nusd"] == 36_000
    assert all(r["metadata"]["trial"] == 1 for r in audit["requests"])


@pytest.mark.parametrize(
    "status, usage",
    [
        (429, {}),
        (500, {}),
        (422, {}),
        (200, {}),
        (200, {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}),
        (200, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 99}),
    ],
)
def test_actual_sdk_error_or_unknown_usage_retains_and_blocks_resend(installed, status, usage):
    import httpx

    guard, sent = installed

    def handle(request):
        sent.append(json.loads(request.content))
        return response(request, status=status, usage=usage)

    guard.transport_factory = lambda: httpx.MockTransport(handle)
    token = gc.TASK.set(("task0", 0))
    try:
        with pytest.raises(Exception):
            user_turn()
        with pytest.raises(Exception):
            user_turn()
    finally:
        gc.TASK.reset(token)
    assert len(sent) == 1
    assert guard.ledger.summary() == {"calls": 1, "allocated_nusd": 12_000, "halted": True}
    assert guard.ledger.audit()["unknown"] == 1


def test_actual_sdk_internal_retry_cannot_forward_unreserved_attempt(installed, monkeypatch):
    import httpx
    from litellm.llms.openai.openai import OpenAIChatCompletion

    guard, sent = installed
    # LiteLLM sets max_retries on the supplied client. Inject after that
    # actual mutation so this causal test really exercises SDK retries.
    original = OpenAIChatCompletion._set_dynamic_params_on_client

    def legacy_retry_client(self, client, **kwargs):
        original(self, client, **kwargs)
        client.max_retries = 2

    monkeypatch.setattr(OpenAIChatCompletion, "_set_dynamic_params_on_client", legacy_retry_client)
    # Deliberately re-enable the exact SDK's retry layer, but not any network.
    from openai._base_client import SyncAPIClient

    sdk_retries = []
    monkeypatch.setattr(
        SyncAPIClient, "_sleep_for_retry", lambda *a, **kw: sdk_retries.append(True)
    )

    def handle(request):
        sent.append(json.loads(request.content))
        raise httpx.ReadTimeout("offline response lost", request=request)

    guard.transport_factory = lambda: httpx.MockTransport(handle)
    token = gc.TASK.set(("task0", 0))
    try:
        with pytest.raises(Exception):
            user_turn(num_retries=3)
    finally:
        gc.TASK.reset(token)
    assert len(sent) == 1
    assert guard.ledger.audit()["unknown"] == 1 and guard.ledger.summary()["halted"]
    assert len(sdk_retries) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"stream": True},
        {"n": 2},
        {"max_tokens": 999},
        {"extra_body": {"max_completion_tokens": 999}},
        {"fallbacks": ["paid/other"]},
    ],
)
def test_actual_source_call_rejects_alternate_paid_route_before_send(installed, overrides):
    guard, sent = installed
    token = gc.TASK.set(("task0", 0))
    try:
        with pytest.raises(gc.BudgetError):
            user_turn(**overrides)
    finally:
        gc.TASK.reset(token)
    assert not sent and guard.ledger.summary()["calls"] == 0


def test_unknown_direct_openai_sdk_client_has_zero_send(installed):
    import openai

    guard, sent = installed
    with openai.OpenAI(api_key="offline-test-key", max_retries=0) as client:
        with pytest.raises(openai.APIConnectionError):
            client.chat.completions.create(
                model="gpt-5.2", messages=[{"role": "user", "content": "offline"}]
            )
    assert not sent and guard.ledger.summary()["halted"]


def test_missing_task_context_denies_known_user_source(installed):
    guard, sent = installed
    with pytest.raises(Exception):
        user_turn()
    assert not sent and guard.ledger.summary()["calls"] == 0


def test_policy_requires_both_models_and_independent_provider_context(configured):
    policy_path, context_path, digest, _, _ = configured
    data = json.loads(policy_path.read_text())
    data["models"].pop("openai/gpt-4.1-2025-04-14")
    policy_path.write_text(json.dumps(data))
    with pytest.raises(gc.BudgetError, match="Both user"):
        gc.load_policy(policy_path, context_path, digest)
    with pytest.raises(gc.BudgetError, match="evidence hash"):
        gc.load_policy(policy_path, context_path, "0" * 64)


def test_unqualified_sdk_bootstrap_never_calls_tau2_main(configured, monkeypatch):
    policy_path, context_path, digest, ledger, _ = configured

    def unqualified():
        raise gc.BudgetError("fixed SDK source mismatch")

    monkeypatch.setattr(gc, "verify_sources", unqualified)
    with pytest.raises(gc.BudgetError, match="source mismatch"):
        gc.main(
            [
                "--policy",
                str(policy_path),
                "--contexts",
                str(context_path),
                "--contexts-sha256",
                digest,
                "--ledger",
                str(ledger.path),
                "--scope",
                ledger.policy.scope_id,
                "--check",
            ]
        )
    assert ledger.summary()["calls"] == 0


def test_explicit_cli_runs_real_batch_context_and_both_sdk_roles(configured, monkeypatch, request):
    import httpx
    from tau2 import cli
    from tau2.runner import batch

    policy_path, context_path, digest, ledger, _ = configured
    sent = []

    def handle(request):
        data = json.loads(request.content)
        sent.append(data)
        content = (
            "hello"
            if data["model"] == "gpt-5.2"
            else json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "offline assertion",
                            "metExpectation": True,
                            "reasoning": "offline",
                        }
                    ]
                }
            )
        )
        return response(request, content)

    real_init = gc.Guard.__init__
    created_guards = []
    request.addfinalizer(lambda: [guard.close() for guard in created_guards])

    def init(self, *args, **kwargs):
        real_init(
            self, *args, **(kwargs | {"transport_factory": lambda: httpx.MockTransport(handle)})
        )
        created_guards.append(self)

    monkeypatch.setattr(gc.Guard, "__init__", init)

    def simulation(*args, **kwargs):
        assert user_turn().content == "hello"
        assert judge_turn()[0].met
        return types.SimpleNamespace(ticks=None)

    monkeypatch.setattr(batch, "run_single_task", simulation)

    def main():
        assert sys.argv[-4:] == ["--max-retries", "0", "--hallucination-retries", "0"]
        ctx = types.SimpleNamespace(
            config=types.SimpleNamespace(
                hallucination_retries=0,
                max_retries=0,
                retry_delay=0,
                verbose_logs=False,
                auto_review=False,
                review_mode="full",
                review_model="unused",
            ),
            save_dir=None,
            monitor=None,
            shutdown_event=None,
            llm_log_mode_value=None,
            evaluation_type=None,
            user_voice_settings=None,
            user_persona_config=None,
            console_display=False,
            save_fn=None,
        )
        # Actual pinned run_unit, run_with_retry, source aliases and both SDKs;
        # only the simulation's model interaction and HTTP transport are fake.
        assert batch.run_unit(ctx, types.SimpleNamespace(id="retail0"), 1, 300).trial == 1

    monkeypatch.setattr(cli, "main", main)
    assert (
        gc.main(
            [
                "--policy",
                str(policy_path),
                "--contexts",
                str(context_path),
                "--contexts-sha256",
                digest,
                "--ledger",
                str(ledger.path),
                "--scope",
                ledger.policy.scope_id,
                "--arm",
                "vllm",
                "--domain",
                "retail",
                "--invocation",
                "real-entry",
                "--local-model",
                "local-model",
                "--local-url",
                "http://127.0.0.1:8000",
                "--",
                "run",
                "--domain",
                "retail",
                "--agent",
                "llm_agent",
                "--user-llm",
                "gpt-5.2",
                "--task-split-name",
                "base",
            ]
        )
        == 0
    )
    assert all(guard.patches for guard in created_guards)  # survives main return
    assert len(sent) == 2 and all(body["service_tier"] == "default" for body in sent)
    records = ledger.audit()["requests"]
    assert {r["metadata"]["role"] for r in records} == {"user", "nl_judge"}
    assert all(
        r["metadata"]["invocation_id"] == "real-entry" and r["metadata"]["task_id"] == "retail0"
        for r in records
    )


@pytest.mark.parametrize("field", ["service_tier", "max_completion_tokens"])
def test_actual_sdk_dropped_protected_wire_field_zero_send(installed, monkeypatch, field):
    from litellm.llms.openai.openai import OpenAIChatCompletion

    guard, sent = installed
    original = OpenAIChatCompletion.make_sync_openai_chat_completion_request

    def drop(self, **kwargs):
        kwargs["data"].pop(field, None)
        return original(self, **kwargs)

    monkeypatch.setattr(OpenAIChatCompletion, "make_sync_openai_chat_completion_request", drop)
    token = gc.TASK.set(("task0", 0))
    try:
        with pytest.raises(Exception):
            user_turn()
    finally:
        gc.TASK.reset(token)
    assert not sent and guard.ledger.summary()["calls"] == 0


@pytest.mark.parametrize("tier", [None, "flex", "priority"])
def test_unknown_actual_service_tier_retains_and_halts(installed, tier):
    import httpx

    guard, sent = installed

    def handle(request):
        sent.append(json.loads(request.content))
        body = response(request).json()
        body["service_tier"] = tier
        return httpx.Response(200, json=body)

    guard.transport_factory = lambda: httpx.MockTransport(handle)
    token = gc.TASK.set(("task0", 0))
    try:
        with pytest.raises(Exception):
            user_turn()
    finally:
        gc.TASK.reset(token)
    assert sent[0]["service_tier"] == "default"
    assert guard.ledger.summary()["halted"] and guard.ledger.audit()["unknown"] == 1
    assert guard.ledger.audit()["requests"][0]["metadata"]["service_tier"] == tier


def sdk_contender(policy_path, context_path, digest, ledger_path, arm, role, ready, sent):
    import os

    os.environ["OPENAI_API_KEY"] = "offline-test-key"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["LITELLM_TELEMETRY"] = "False"

    def forbidden(*args, **kwargs):
        raise AssertionError("real network forbidden in child")

    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    import httpx

    policy, contexts = gc.load_policy(policy_path, context_path, digest)
    ledger = gc.Ledger(ledger_path, policy, create=False)

    def handle(request):
        sent.put((arm, json.loads(request.content)["model"]))
        content = (
            "hello"
            if role == "user"
            else json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "offline assertion",
                            "metExpectation": True,
                            "reasoning": "offline",
                        }
                    ]
                }
            )
        )
        return response(request, content)

    guard = gc.Guard(
        ledger,
        contexts,
        arm=arm,
        domain="retail",
        invocation_id=arm,
        local_model="local-model",
        local_url="http://127.0.0.1:8000",
        transport_factory=lambda: httpx.MockTransport(handle),
    )
    try:
        guard.install()
        ready.wait(timeout=20)
        token = gc.TASK.set(("task0", 0))
        try:
            (user_turn if role == "user" else judge_turn)()
        except Exception:
            pass  # the losing actual SDK request is refused before fake transport
        finally:
            gc.TASK.reset(token)
    finally:
        guard.close()


def test_two_actual_sdk_processes_two_models_share_last_global_slot(configured):
    import multiprocessing
    import queue

    policy_path, context_path, digest, _, _ = configured
    data = json.loads(policy_path.read_text())
    data["cap_nusd"] = 24_000  # user12k or judge24k; never BOTH36k
    policy_path.write_text(json.dumps(data))
    policy, _ = gc.load_policy(policy_path, context_path, digest)
    ledger = gc.Ledger(policy_path.parent / "shared-near-cap.db", policy)
    ctx = multiprocessing.get_context("spawn")
    ready, sent = ctx.Barrier(2), ctx.Queue()
    workers = [
        ctx.Process(
            target=sdk_contender,
            args=(policy_path, context_path, digest, ledger.path, arm, role, ready, sent),
        )
        for arm, role in [("vllm", "user"), ("smg", "nl_judge")]
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(35)
            assert worker.exitcode == 0
        assert sent.get(timeout=1) in [("vllm", "gpt-5.2"), ("smg", "gpt-4.1-2025-04-14")]
        with pytest.raises(queue.Empty):
            sent.get(timeout=0.2)
        assert ledger.summary()["calls"] == 1 and ledger.summary()["allocated_nusd"] <= 24_000
        assert ledger.audit()["pending"] == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        sent.close()


def test_actual_async_openai_alternate_path_zero_send(installed):
    import asyncio

    import openai

    guard, sent = installed

    async def attempt():
        async with openai.AsyncOpenAI(api_key="offline-test-key", max_retries=0) as client:
            with pytest.raises(openai.APIConnectionError):
                await client.chat.completions.create(
                    model="gpt-5.2", messages=[{"role": "user", "content": "offline"}]
                )

    asyncio.run(attempt())
    assert not sent and guard.ledger.summary()["calls"] == 0 and guard.ledger.summary()["halted"]


@pytest.mark.parametrize("failure", [400, 500, "timeout"])
def test_local_sdk_failure_does_not_halt_paid_budget(installed, monkeypatch, failure):
    import httpx
    from tau2.data_model.message import UserMessage
    from tau2.utils import llm_utils

    guard, sent = installed
    visited = []

    def handle(request):
        visited.append(str(request.url))
        if failure == "timeout":
            raise httpx.ReadTimeout("offline local timeout", request=request)
        return response(request, status=failure)

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", lambda _, request: transport.handle_request(request)
    )
    token = gc.TASK.set(("task-local-failure", 0))
    try:
        with pytest.raises(Exception):
            llm_utils.generate(
                model="openai/local-model",
                messages=[UserMessage(role="user", content="offline")],
                api_base="http://127.0.0.1:8000/v1",
                api_key="smg-local",
                num_retries=0,
                max_retries=0,
            )
        assert visited and all(httpx.URL(url).host == "127.0.0.1" for url in visited)
        assert not sent and guard.ledger.summary()["calls"] == 0
        assert not guard.failed and not guard.ledger.summary()["halted"]
        assert user_turn().content == "hello"
        assert len(sent) == 1 and guard.ledger.summary()["calls"] == 1
        assert not guard.failed and not guard.ledger.summary()["halted"]
    finally:
        gc.TASK.reset(token)


def test_actual_local_sdk_redirect_cannot_become_free_external_send(installed, monkeypatch):
    import httpx
    from tau2.data_model.message import UserMessage
    from tau2.utils import llm_utils

    guard, sent = installed
    visited = []

    def local_redirect(self, request):
        visited.append(str(request.url))
        if request.url.host == "127.0.0.1":
            return httpx.Response(
                307,
                headers={"location": "https://api.openai.com/v1/chat/completions"},
                json={"error": "offline redirect"},
            )
        return response(request)  # old unsafe redirect would reach this branch

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", local_redirect)
    with pytest.raises(Exception):
        llm_utils.generate(
            model="openai/local-model",
            messages=[UserMessage(role="user", content="offline")],
            api_base="http://127.0.0.1:8000/v1",
            api_key="smg-local",
            num_retries=0,
            max_retries=0,
        )
    assert visited and all(httpx.URL(url).host == "127.0.0.1" for url in visited)
    assert not sent and guard.ledger.summary()["calls"] == 0 and guard.ledger.summary()["halted"]


def test_real_source_fingerprint_mismatch_refuses_bootstrap(configured, monkeypatch):
    policy_path, context_path, digest, ledger, _ = configured
    monkeypatch.setitem(gc.TAU2_SHA, "utils/llm_utils.py", "0" * 64)
    with pytest.raises(gc.BudgetError, match="Unqualified tau2 source"):
        gc.main(
            [
                "--policy",
                str(policy_path),
                "--contexts",
                str(context_path),
                "--contexts-sha256",
                digest,
                "--ledger",
                str(ledger.path),
                "--scope",
                ledger.policy.scope_id,
                "--check",
            ]
        )
    assert ledger.summary()["calls"] == 0


def test_editable_package_source_initializes_and_rejects_changed_bytes(configured, tmp_path):
    import importlib.metadata
    import importlib.util

    policy, contexts, digest, _, _ = configured
    environment = tmp_path / "editable-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    site = environment / "lib/python3.12/site-packages"
    source = tmp_path / "editable-source/tau2"
    source.mkdir(parents=True)
    actual = Path(importlib.util.find_spec("tau2").origin).parent
    for name in ["__init__.py", *gc.TAU2_SHA]:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual / name, target)
    # Real interpreter startup consumes the PEP660 .pth layout, while the
    # distribution metadata lives in site-packages without a tau2/ directory.
    distribution = importlib.metadata.distribution("tau2")
    metadata = site / "tau2-1.0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(distribution.read_text("METADATA"))
    (metadata / "direct_url.json").write_text(
        json.dumps({"url": source.parent.as_uri(), "dir_info": {"editable": True}})
    )
    (site / "_tau2.pth").write_text(str(source.parent) + "\n")
    (site / "zz_qualified_sdks.pth").write_text(
        str(Path(importlib.metadata.distribution("openai").locate_file(""))) + "\n"
    )
    ledger = tmp_path / "editable-new.db"
    command = [
        str(environment / "bin/python"),
        "-I",
        gc.__file__,
        "--policy",
        str(policy),
        "--contexts",
        str(contexts),
        "--contexts-sha256",
        digest,
        "--ledger",
        str(ledger),
        "--scope",
        "offline-two-model-scope",
    ]
    result = subprocess.run(command + ["--initialize"], capture_output=True, text=True, timeout=20)
    assert not (site / "tau2").exists()
    assert result.returncode == 0, result.stderr
    assert ledger.is_file()
    changed = source / "utils/llm_utils.py"
    changed.write_bytes(changed.read_bytes() + b"\n# changed fixture\n")
    rejected = subprocess.run(command + ["--check"], capture_output=True, text=True, timeout=20)
    assert rejected.returncode == 2
    assert "Unqualified tau2 source: utils/llm_utils.py" in rejected.stderr


@pytest.mark.parametrize("parallel, fail", [(False, False), (True, False), (False, True)])
def test_actual_sdk_client_handles_bound_to_inflight_and_close_on_error(installed, parallel, fail):
    import concurrent.futures
    import threading

    import httpx

    guard, _ = installed
    lock = threading.Lock()
    counts = {"active": 0, "peak": 0, "closed": 0}
    workers = 4 if parallel else 1
    barrier = threading.Barrier(workers)

    class TrackedTransport(httpx.BaseTransport):
        def __init__(self):
            with lock:
                counts["active"] += 1
                counts["peak"] = max(counts["peak"], counts["active"])

        def handle_request(self, request):
            if parallel:
                barrier.wait(timeout=10)
            return response(request, status=429 if fail else 200)

        def close(self):
            with lock:
                counts["active"] -= 1
                counts["closed"] += 1

    guard.transport_factory = TrackedTransport

    def call(index):
        token = gc.TASK.set(("task" + str(index), 0))
        try:
            if fail:
                with pytest.raises(Exception):
                    user_turn()
            else:
                assert user_turn().content == "hello"
        finally:
            gc.TASK.reset(token)

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(call, range(workers)))
        expected = workers
    else:
        expected = 1 if fail else 5
        for index in range(expected):
            call(index)
            assert not guard.clients
    assert counts["active"] == 0 and counts["closed"] == expected
    assert counts["peak"] <= workers and not guard.clients


def test_actual_litellm_semantic_retry_needs_a_second_reservation(installed, monkeypatch):
    from dataclasses import replace

    from litellm.llms.openai.openai import OpenAIChatCompletion

    guard, sent = installed
    policy = replace(guard.ledger.policy, cap_nusd=12_000, scope_id="offline-semantic-retry")
    guard.ledger = gc.Ledger(guard.ledger.path.with_name("semantic-retry.db"), policy)
    original = OpenAIChatCompletion.make_sync_openai_chat_completion_request
    attempts = []

    def format_fault(self, **kwargs):
        attempts.append(True)
        result = original(self, **kwargs)
        if len(attempts) == 1:
            # Inject at the exact SDK helper return boundary. The pinned
            # LiteLLM range(2) reformat loop, NOT our adapter, retries this.
            raise ValueError("Conversation roles must alternate user/assistant")
        return result

    monkeypatch.setattr(
        OpenAIChatCompletion, "make_sync_openai_chat_completion_request", format_fault
    )
    token = gc.TASK.set(("task0", 0))
    try:
        with pytest.raises(Exception):
            user_turn()
    finally:
        gc.TASK.reset(token)
    assert len(attempts) == 2 and len(sent) == 1
    assert guard.ledger.summary() == {"calls": 1, "allocated_nusd": 12_000, "halted": True}


@pytest.mark.parametrize(
    "extra", [["--domain", "banking_knowledge"], ["--auto-review"], ["--task-split-name", "test"]]
)
def test_cli_cannot_override_fixed_text_scope(extra):
    command = [
        "run",
        "--domain",
        "retail",
        "--agent",
        "llm_agent",
        "--user-llm",
        "gpt-5.2",
        "--task-split-name",
        "base",
    ]
    with pytest.raises(gc.BudgetError):
        gc.validate_command(command + extra, "retail")


def test_one_time_initializer_refuses_ledger_replacement(configured):
    policy_path, context_path, digest, ledger, _ = configured
    argv = [
        "--policy",
        str(policy_path),
        "--contexts",
        str(context_path),
        "--contexts-sha256",
        digest,
        "--ledger",
        str(ledger.path),
        "--scope",
        ledger.policy.scope_id,
        "--initialize",
    ]
    old = ledger.path.read_bytes()
    with pytest.raises(FileExistsError):
        gc.main(argv)
    assert ledger.path.read_bytes() == old
    fresh = ledger.path.with_name("explicit-new-scope.db")
    data = json.loads(policy_path.read_text())
    data["scope_id"] = "separately-approved-new-scope"
    policy_path.write_text(json.dumps(data))
    argv[argv.index("--scope") + 1] = data["scope_id"]
    new_policy, _ = gc.load_policy(policy_path, context_path, digest)
    argv[argv.index("--ledger") + 1] = str(fresh)
    assert gc.main(argv) == 0
    assert gc.Ledger(fresh, new_policy, create=False).summary()["calls"] == 0


def standing_fixture():
    return {
        "cap_nusd": 100,
        "calls": 2,
        "calls_per_arm": 1,
        "approval": {
            "mode": "standing",
            "budget_unit": "per_comparison_job",
            "workflow_ref": "org/repo/.github/workflows/nightly-tau2.yml@refs/heads/main",
            "allowed_events": ["schedule"],
            "matrix_names": ["small", "large"],
            "not_before": "2026-11-09T00:00:00Z",
            "expires_at": "2026-11-11T00:00:00Z",
        },
    }


def standing_context(run="123", name="small"):
    return {
        "run_id": run,
        "run_attempt": "1",
        "matrix_name": name,
        "workflow_ref": standing_fixture()["approval"]["workflow_ref"],
        "event_name": "schedule",
    }


def test_standing_allowance_reuses_explicit_caps_for_each_approved_job():
    from datetime import datetime

    now = datetime(2026, 11, 10, tzinfo=UTC)
    template = standing_fixture()
    policies = [
        gc.resolve_approval(template, standing_context(run, name), now)
        for run in ("123", "124")
        for name in ("small", "large")
    ]
    assert len({p["scope_id"] for p in policies}) == 4
    assert all(
        p == {"cap_nusd": 100, "calls": 2, "calls_per_arm": 1, "scope_id": p["scope_id"]}
        for p in policies
    )
    assert "scope_id" not in template  # approval template stays immutable


@pytest.mark.parametrize(
    "change",
    [
        {"workflow_ref": "other/ref"},
        {"event_name": "pull_request"},
        {"matrix_name": "unapproved"},
        {"run_id": ""},
    ],
)
def test_standing_context_mismatch_refuses_before_initialization(change, monkeypatch):
    from datetime import datetime

    context = standing_context() | change
    monkeypatch.setattr(gc.os, "open", lambda *a, **k: pytest.fail("must not initialize"))
    with pytest.raises(gc.BudgetError):
        gc.resolve_approval(standing_fixture(), context, datetime(2026, 11, 10, tzinfo=UTC))


@pytest.mark.parametrize("day", [8, 11])
def test_standing_start_window_denies_early_or_expired(day):
    from datetime import datetime

    with pytest.raises(gc.BudgetError):
        gc.resolve_approval(
            standing_fixture(), standing_context(), datetime(2026, 11, day, tzinfo=UTC)
        )


def test_exact_job_mode_stays_exact_and_cannot_mix_modes():
    from datetime import datetime

    now = datetime(2026, 11, 10, tzinfo=UTC)
    assert (
        gc.resolve_approval({"scope_id": "123:1:small"}, standing_context(), now)["scope_id"]
        == "123:1:small"
    )
    with pytest.raises(gc.BudgetError):
        gc.resolve_approval({"scope_id": "123:1:small"}, standing_context("124"), now)
    with pytest.raises(gc.BudgetError):
        gc.resolve_approval(
            standing_fixture() | {"scope_id": "123:1:small"}, standing_context(), now
        )


def test_isolated_guarded_entry_does_not_activate_legacy_startup_hook(configured):
    policy_path, context_path, digest, ledger, _ = configured
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(gc.__file__).parent / "costmap_shim")
    # A legacy shim import would reject this before even importing LiteLLM.
    env["SMG_TAU2_BUDGET_GUARD"] = "invalid"
    command = [
        sys.executable,
        "-I",
        gc.__file__,
        "--policy",
        str(policy_path),
        "--contexts",
        str(context_path),
        "--contexts-sha256",
        digest,
        "--ledger",
        str(ledger.path),
        "--scope",
        ledger.policy.scope_id,
        "--check",
    ]
    isolated = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert isolated.returncode == 0, isolated.stderr
    # Causal control: identical environment and entry, removing only -I, is
    # rejected by the preserved real legacy startup module before main runs.
    legacy = subprocess.run(
        [part for part in command if part != "-I"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert legacy.returncode != 0
    assert "SMG_TAU2_BUDGET_GUARD must be 0 or 1" in legacy.stderr
    assert ledger.summary()["calls"] == 0
