"""One servicer switch for the four launchers (``rust_lifecycle.servicer_switch``):
the flag, else the engine's variable, else the launcher's default (rust for vLLM,
SGLang and TokenSpeed; python still for the TensorRT-LLM engine), and only a
flag decision is written back to the environment the engine processes inherit;
on a launcher whose default is rust an explicit python choice is announced with
the deprecation line, the default with none."""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace

import pytest
from smg_grpc_servicer import rust_lifecycle
from smg_grpc_servicer.sglang import plugin as sglang_plugin
from smg_grpc_servicer.sglang import rust as sglang_rust
from smg_grpc_servicer.tokenspeed import rust as tokenspeed_rust
from smg_grpc_servicer.trtllm import rust as trtllm_rust
from smg_grpc_servicer.vllm import rust as vllm_rust

# Each launcher with the implementation it runs when nothing selects one.
LAUNCHERS = [
    pytest.param(vllm_rust, "rust", id="vllm"),
    pytest.param(sglang_rust, "rust", id="sglang"),
    pytest.param(tokenspeed_rust, "rust", id="tokenspeed"),
    pytest.param(trtllm_rust, "python", id="trtllm"),
]


@pytest.fixture(autouse=True)
def no_parsed_flag(monkeypatch):
    """No ``--servicer-impl`` parsed by the SGLang plugin in this process."""
    monkeypatch.setattr(sglang_plugin._state, "parsed", None)


@pytest.mark.parametrize(("launcher", "default"), LAUNCHERS)
def test_the_switch_reads_the_flag_then_the_variable_then_the_launchers_default(launcher, default):
    env = launcher.SERVICER_IMPL_ENV
    assert launcher.resolve_servicer_impl(environ={}) == default
    assert launcher.resolve_servicer_impl(environ={env: " Python "}) == "python"
    assert launcher.resolve_servicer_impl(environ={env: " Rust "}) == "rust"
    rust = argparse.Namespace(servicer_impl="rust")
    python = argparse.Namespace(servicer_impl="python")
    assert launcher.resolve_servicer_impl(rust, environ={}) == "rust"
    assert launcher.resolve_servicer_impl(python, environ={}) == "python"
    # The flag wins over the environment, in both directions.
    assert launcher.resolve_servicer_impl(rust, environ={env: "python"}) == "rust"
    assert launcher.resolve_servicer_impl(python, environ={env: "rust"}) == "python"
    # An unset flag falls through to the environment, then to the default.
    unset = argparse.Namespace(servicer_impl=None)
    assert launcher.resolve_servicer_impl(unset, environ={env: "rust"}) == "rust"
    assert launcher.resolve_servicer_impl(unset, environ={}) == default
    with pytest.raises(rust_lifecycle.ServicerConfigError, match=f"{env} must be one of"):
        launcher.resolve_servicer_impl(environ={env: "go"})
    with pytest.raises(rust_lifecycle.ServicerConfigError, match=f"{env} must be one of"):
        launcher.resolve_servicer_impl(argparse.Namespace(servicer_impl="go"), environ={})


@pytest.mark.parametrize(("launcher", "default"), LAUNCHERS)
def test_only_a_flag_decision_is_written_back_to_the_environment(launcher, default):
    env = launcher.SERVICER_IMPL_ENV
    # `--servicer-impl python` with `rust` exported: the flag decides, and the
    # environment the engine processes inherit says the same.
    environ = {env: "rust"}
    python = argparse.Namespace(servicer_impl="python")
    assert launcher.resolve_servicer_impl(python, environ) == "python"
    assert environ == {env: "python"}
    # Without a flag the environment stays exactly as the operator exported
    # it: nothing when nothing was (the default is not written), their
    # spelling otherwise, whether or not the launcher's namespace is in hand.
    for exported in ({}, {env: "rust"}, {env: " Rust "}, {env: "python"}):
        for args in (None, SimpleNamespace(), argparse.Namespace(servicer_impl=None)):
            environ = dict(exported)
            launcher.resolve_servicer_impl(args, environ)
            assert environ == exported, (args, exported)


def test_the_launchers_answer_with_the_shared_switch():
    """Every launcher's ``servicer_impl_source`` is the shared switch with the
    launcher's own default: the origin names the default, the variable or the
    flag, and the choices are one tuple."""
    for launcher, default in ((sglang_rust, "rust"), (tokenspeed_rust, "rust")):
        env = launcher.SERVICER_IMPL_ENV
        other = "python" if default == "rust" else "rust"
        assert launcher.servicer_impl_source(environ={}) == (default, "default")
        assert launcher.servicer_impl_source(environ={env: other}) == (other, "env")
        flag = argparse.Namespace(servicer_impl=other)
        assert launcher.servicer_impl_source(flag, environ={}) == (other, "flag")
    assert vllm_rust.servicer_impl_source(environ={}) == ("rust", "default")
    assert trtllm_rust.servicer_impl_source(environ={}) == ("python", "default")
    assert (
        vllm_rust.IMPLS
        is sglang_rust.IMPLS
        is tokenspeed_rust.IMPLS
        is trtllm_rust.IMPLS
        is rust_lifecycle.IMPLS
    )


def test_the_switchs_own_default_is_rust_and_a_launcher_may_pass_another():
    assert rust_lifecycle.DEFAULT_IMPL == "rust"
    assert rust_lifecycle.servicer_switch("X_SERVICER_IMPL", environ={}) == ("rust", "default")
    assert rust_lifecycle.servicer_switch("X_SERVICER_IMPL", environ={}, default="python") == (
        "python",
        "default",
    )
    # The default is not a value: a launcher cannot pass one outside the choices.
    with pytest.raises(rust_lifecycle.ServicerConfigError, match="X_SERVICER_IMPL must be one of"):
        rust_lifecycle.servicer_switch("X_SERVICER_IMPL", environ={}, default="go")


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


def test_an_explicit_python_choice_is_announced_as_deprecated_where_rust_is_the_default(caplog):
    """The start-up line names the implementation and its origin; on a launcher
    whose default is rust the deprecation warning follows a python chosen by the
    flag or the variable (the default never selects it, and an explicit rust logs
    no warning); a launcher whose default is still python logs no deprecation."""
    log = logging.getLogger("test.servicer.announce")
    with caplog.at_level(logging.INFO, logger=log.name):
        rust_lifecycle.announce_servicer_impl("rust", "default", "X_SERVICER_IMPL", log)
        rust_lifecycle.announce_servicer_impl("rust", "flag", "X_SERVICER_IMPL", log)
        rust_lifecycle.announce_servicer_impl(
            "python", "default", "X_SERVICER_IMPL", log, default="python"
        )
        rust_lifecycle.announce_servicer_impl(
            "rust", "env", "X_SERVICER_IMPL", log, default="python"
        )
    assert [r.levelno for r in caplog.records] == [logging.INFO] * 4
    assert caplog.records[0].getMessage() == "Servicer implementation: rust (source=default)"
    assert caplog.records[2].getMessage() == "Servicer implementation: python (source=default)"
    caplog.clear()
    for origin in ("flag", "env"):
        with caplog.at_level(logging.INFO, logger=log.name):
            rust_lifecycle.announce_servicer_impl("python", origin, "X_SERVICER_IMPL", log)
        assert [r.levelno for r in caplog.records] == [logging.INFO, logging.WARNING], origin
        warning = caplog.records[1].getMessage()
        assert warning.startswith("The Python gRPC servicer is deprecated")
        assert "X_SERVICER_IMPL=python" in warning and "--servicer-impl python" in warning
        assert "later release" in warning
        caplog.clear()
