import ast
import hashlib
import importlib
import importlib.util
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

TAU2 = Path(__file__).resolve().parents[1] / "tau2"
# Full MIT source: sierra-research/tau2-bench@5bfa7e37b36656b37dc6d022156be6563c1007f3.
# CI checks out that revision; local offline runs can provide its verified file.
SOURCE = Path(
    os.environ.get(
        "TAU2_LLM_UTILS_SOURCE",
        Path(__file__).resolve().parents[2] / ".tau2-upstream/src/tau2/utils/llm_utils.py",
    )
)


def pinned_source():
    source = SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "fa7d61e34ef9c668452ec44e8ba9efafec14a349a530481fac1dc19eff275c65"
    )
    return source.decode()


def generate(litellm, monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    full_source = pinned_source()
    node = next(n for n in ast.parse(full_source).body if getattr(n, "name", None) == "generate")
    source = "".join(full_source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
    assert hashlib.sha256(source.encode()).hexdigest() == (
        "9496cc4c9571388354d8f883785978f9e89f2da427961d019cf69c6b7b112be2"
    )

    def noop(*args, **kwargs):
        pass

    namespace = dict(
        os=os,
        json=json,
        time=time,
        datetime=datetime,
        DEFAULT_MAX_RETRIES=2,
        validate_message_history=noop,
        to_litellm_messages=lambda v: v,
        _format_messages_for_logging=lambda v: v,
        _write_llm_log=noop,
        get_response_cost=lambda r: 0.0,
        get_response_usage=lambda r: r.get("usage"),
        AssistantMessage=lambda **kw: SimpleNamespace(**kw),
        logger=SimpleNamespace(error=noop, warning=noop),
    )
    exec("from __future__ import annotations\nfrom litellm import completion\n" + source, namespace)
    assert namespace["completion"] is litellm.completion
    return namespace["generate"]


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TAU2))
    hook, budget = importlib.import_module("budget_hook"), importlib.import_module("budget")
    ledger = budget.Ledger(
        tmp_path / "budget.db", budget.Policy("paid/model", "1", "1", 4000, 2, 1, 10, 10)
    )
    sent = []

    class Response(dict):
        def __getattr__(self, name):
            return self[name]

        def to_dict(self):
            return dict(self)

    response = Response(
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="ok", tool_calls=None),
            )
        ],
    )

    def provider(**payload):
        sent.append(payload)
        if "error" in response:
            raise response["error"]
        return response

    litellm = SimpleNamespace(completion=provider, drop_params=False, register_model=lambda v: None)
    guarded = hook.install(litellm)
    guarded.authorizer = lambda payload: (ledger, "vllm", 1, "max_completion_tokens", True)
    return SimpleNamespace(
        ledger=ledger,
        sent=sent,
        response=response,
        litellm=litellm,
        guarded=guarded,
        generate=generate(litellm, monkeypatch),
    )


def invoke(c, **kwargs):
    return c.generate(
        "paid/model",
        [{"role": "user", "content": "hello"}],
        num_retries=0,
        max_completion_tokens=1,
        **kwargs,
    )


def test_pinned_generate_reaches_reserved_provider(context):
    assert invoke(context).content == "ok"
    assert len(context.sent) == 1 and context.ledger.summary()["allocated_nusd"] == 2000


@pytest.mark.parametrize(
    "failure", ["no_authorizer", "drop_params", "unknown_limit", "unknown_bound", "unverified"]
)
def test_unknown_capabilities_block_actual_generate_before_send(context, failure):
    if failure == "no_authorizer":
        context.guarded.authorizer = None
    elif failure == "drop_params":
        context.litellm.drop_params = True
    else:
        entry = {
            "unknown_limit": (1, "other", True),
            "unknown_bound": (None, "max_completion_tokens", True),
            "unverified": (1, "max_completion_tokens", False),
        }[failure]
        context.guarded.authorizer = lambda payload: (context.ledger, "vllm", *entry)
    with pytest.raises(ValueError):
        invoke(context)
    assert context.sent == [] and context.ledger.summary()["calls"] == 0


@pytest.mark.parametrize("model,kw", [("paid/model", {}), ("bad/model", {"num_retries": 0})])
def test_default_retries_and_unknown_model_are_rejected(context, model, kw):
    with pytest.raises(ValueError):
        context.generate(model, [], max_completion_tokens=1, **kw)
    assert context.sent == []


def test_timeout_after_pinned_generate_send_halts(context):
    context.response["error"] = TimeoutError("lost response")
    with pytest.raises(ValueError):
        invoke(context)
    assert len(context.sent) == 1 and context.ledger.summary()["halted"]


@pytest.mark.parametrize("usage", [None, "malformed", {"prompt_tokens": 2, "completion_tokens": 1}])
def test_unknown_usage_halts_after_actual_generate_send(context, usage):
    context.response["usage"] = usage
    with pytest.raises(ValueError):
        invoke(context)
    with pytest.raises(ValueError):
        invoke(context)
    assert len(context.sent) == 1 and context.ledger.summary()["halted"]


def test_sitecustomize_installs_unconfigured_hook(context, monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", context.litellm)
    monkeypatch.setenv("SMG_TAU2_BUDGET_GUARD", "1")
    spec = importlib.util.spec_from_file_location(
        "guarded_site", TAU2 / "costmap_shim/sitecustomize.py"
    )
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert context.litellm.completion.authorizer is None
    with pytest.raises(ValueError):
        generate(context.litellm, monkeypatch)("paid/model", [])
    assert context.sent == []


def full_module(context, monkeypatch):
    """Execute every pinned module statement with offline dependency boundaries."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    clients, cache_disabled = [], []
    config = dict.fromkeys(
        "DEFAULT_LLM_CACHE_TYPE REDIS_CACHE_TTL REDIS_CACHE_VERSION REDIS_HOST "
        "REDIS_PASSWORD REDIS_PORT REDIS_PREFIX".split()
    )
    config.update(DEFAULT_MAX_RETRIES=2, LLM_CACHE_ENABLED=False, USE_LANGFUSE=False)
    messages = {
        name: type(name, (), {})
        for name in (
            "AssistantMessage",
            "Message",
            "ParticipantMessageBase",
            "SystemMessage",
            "ToolCall",
            "ToolMessage",
            "UserMessage",
        )
    }
    imports = {
        "httpx": dict(
            Limits=lambda **kw: kw,
            Client=lambda **kw: clients.append(kw),
            AsyncClient=lambda **kw: clients.append(kw),
        ),
        "litellm.caching.caching": dict(Cache=forbidden),
        "litellm.main": dict(ModelResponse=dict, Usage=dict),
        "loguru": dict(logger=SimpleNamespace(info=lambda *a: None, error=lambda *a: None)),
        "tau2.config": config,
        "tau2.data_model.message": messages,
        "tau2.environment.tool": dict(Tool=type("Tool", (), {})),
    }
    for name, attributes in imports.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        context.litellm, "disable_cache", lambda: cache_disabled.append(True), raising=False
    )
    monkeypatch.setattr(context.litellm, "completion_cost", forbidden, raising=False)
    module = ModuleType("pinned_tau2_llm_utils")
    exec(compile(pinned_source(), str(SOURCE), "exec"), vars(module))
    assert len(clients) == 2 and cache_disabled == [True]
    assert module.completion is context.guarded
    assert context.litellm.drop_params is True
    return module


@pytest.mark.parametrize("kwargs", [{}, {"drop_params": False}])
def test_complete_pinned_module_fails_closed_before_authorization(context, monkeypatch, kwargs):
    module = full_module(context, monkeypatch)

    def unexpected_authorizer(payload):
        pytest.fail("Global drop_params=True must be rejected before authorization")

    context.guarded.authorizer = unexpected_authorizer
    with pytest.raises(ValueError, match="Silently dropped provider limits"):
        module.generate("paid/model", [], num_retries=0, max_completion_tokens=1, **kwargs)
    assert context.sent == [] and context.ledger.summary()["calls"] == 0
    assert context.litellm.drop_params is True


def test_complete_module_retains_the_startup_guard_alias(context, monkeypatch):
    module = full_module(context, monkeypatch)
    context.litellm.completion = lambda **kw: pytest.fail("Unprotected replacement reached")
    assert module.completion is context.guarded
    with pytest.raises(ValueError, match="Silently dropped provider limits"):
        module.generate("paid/model", [], num_retries=0, max_completion_tokens=1)
    assert context.sent == [] and context.ledger.summary()["calls"] == 0
