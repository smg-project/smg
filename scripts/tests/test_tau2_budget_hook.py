import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

TAU2 = Path(__file__).resolve().parents[1] / "tau2"
FIXTURE = Path(__file__).parent / "fixtures" / "tau2_generate.py.txt"


def generate(litellm, monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    source = "".join(FIXTURE.read_text().splitlines(keepends=True)[2:])
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
