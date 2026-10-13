"""One servicer switch for the three launchers (``rust_lifecycle.servicer_switch``):
the flag, else the engine's variable, else python, and only a flag decision is
written back to the environment the engine processes inherit."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from smg_grpc_servicer import rust_lifecycle
from smg_grpc_servicer.sglang import plugin as sglang_plugin
from smg_grpc_servicer.sglang import rust as sglang_rust
from smg_grpc_servicer.tokenspeed import rust as tokenspeed_rust
from smg_grpc_servicer.vllm import rust as vllm_rust

LAUNCHERS = [
    pytest.param(vllm_rust, id="vllm"),
    pytest.param(sglang_rust, id="sglang"),
    pytest.param(tokenspeed_rust, id="tokenspeed"),
]


@pytest.fixture(autouse=True)
def no_parsed_flag(monkeypatch):
    """No ``--servicer-impl`` parsed by the SGLang plugin in this process."""
    monkeypatch.setattr(sglang_plugin._state, "parsed", None)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_the_switch_reads_the_flag_then_the_variable_then_python(launcher):
    env = launcher.SERVICER_IMPL_ENV
    assert launcher.resolve_servicer_impl(environ={}) == "python"
    assert launcher.resolve_servicer_impl(environ={env: " Rust "}) == "rust"
    rust = argparse.Namespace(servicer_impl="rust")
    python = argparse.Namespace(servicer_impl="python")
    assert launcher.resolve_servicer_impl(rust, environ={}) == "rust"
    # The flag wins over the environment, in both directions.
    assert launcher.resolve_servicer_impl(rust, environ={env: "python"}) == "rust"
    assert launcher.resolve_servicer_impl(python, environ={env: "rust"}) == "python"
    # An unset flag falls through to the environment.
    unset = argparse.Namespace(servicer_impl=None)
    assert launcher.resolve_servicer_impl(unset, environ={env: "rust"}) == "rust"
    with pytest.raises(rust_lifecycle.ServicerConfigError, match=f"{env} must be one of"):
        launcher.resolve_servicer_impl(environ={env: "go"})
    with pytest.raises(rust_lifecycle.ServicerConfigError, match=f"{env} must be one of"):
        launcher.resolve_servicer_impl(argparse.Namespace(servicer_impl="go"), environ={})


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_only_a_flag_decision_is_written_back_to_the_environment(launcher):
    env = launcher.SERVICER_IMPL_ENV
    # `--servicer-impl python` with `rust` exported: the flag decides, and the
    # environment the engine processes inherit says the same.
    environ = {env: "rust"}
    python = argparse.Namespace(servicer_impl="python")
    assert launcher.resolve_servicer_impl(python, environ) == "python"
    assert environ == {env: "python"}
    # Without a flag the environment stays exactly as the operator exported
    # it: nothing when nothing was, their spelling otherwise, whether or not
    # the launcher's namespace is in hand.
    for exported in ({}, {env: "rust"}, {env: " Rust "}):
        for args in (None, SimpleNamespace(), argparse.Namespace(servicer_impl=None)):
            environ = dict(exported)
            launcher.resolve_servicer_impl(args, environ)
            assert environ == exported, (args, exported)


def test_the_launchers_answer_with_the_shared_switch():
    for launcher in (sglang_rust, tokenspeed_rust):
        env = launcher.SERVICER_IMPL_ENV
        assert launcher.servicer_impl_source(environ={}) == ("python", "default")
        assert launcher.servicer_impl_source(environ={env: "rust"}) == ("rust", "env")
        flag = argparse.Namespace(servicer_impl="rust")
        assert launcher.servicer_impl_source(flag, environ={}) == ("rust", "flag")
    assert vllm_rust.IMPLS is sglang_rust.IMPLS is tokenspeed_rust.IMPLS is rust_lifecycle.IMPLS


def test_a_flag_the_plugin_parsed_counts_as_the_launchers():
    """A launcher whose namespace carries only the engine's own fields (SGLang's
    ServerArgs) hands over what its plugin parsed; that is the flag, written
    back like one, and the namespace's own flag comes first."""
    environ = {"X_SERVICER_IMPL": "python"}
    assert rust_lifecycle.servicer_switch(
        "X_SERVICER_IMPL", SimpleNamespace(), environ, parsed_flag="rust"
    ) == ("rust", "flag")
    assert environ == {"X_SERVICER_IMPL": "rust"}
    assert rust_lifecycle.servicer_switch(
        "X_SERVICER_IMPL", SimpleNamespace(servicer_impl="python"), {}, parsed_flag="rust"
    ) == ("python", "flag")
