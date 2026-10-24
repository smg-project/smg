"""One servicer switch for the three launchers (``rust_lifecycle.servicer_switch``):
the flag, else the engine's variable, else rust; a launch on defaults that the
Rust path does not serve falls back to python with one line naming the gap; a
flag or fallback decision is written back to the environment the engine
processes inherit; every start of the Python servicer announces the
deprecation."""

from __future__ import annotations

import argparse
import logging
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
def test_the_switch_reads_the_flag_then_the_variable_then_rust(launcher):
    env = launcher.SERVICER_IMPL_ENV
    assert launcher.resolve_servicer_impl(environ={}) == "rust"
    assert launcher.resolve_servicer_impl(environ={env: " Python "}) == "python"
    assert launcher.resolve_servicer_impl(environ={env: " Rust "}) == "rust"
    rust = argparse.Namespace(servicer_impl="rust")
    python = argparse.Namespace(servicer_impl="python")
    assert launcher.resolve_servicer_impl(python, environ={}) == "python"
    # The flag wins over the environment, in both directions.
    assert launcher.resolve_servicer_impl(rust, environ={env: "python"}) == "rust"
    assert launcher.resolve_servicer_impl(python, environ={env: "rust"}) == "python"
    # An unset flag falls through to the environment, then to the default.
    unset = argparse.Namespace(servicer_impl=None)
    assert launcher.resolve_servicer_impl(unset, environ={env: "python"}) == "python"
    assert launcher.resolve_servicer_impl(unset, environ={}) == "rust"
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
    for exported in ({}, {env: "rust"}, {env: " Rust "}, {env: "python"}):
        for args in (None, SimpleNamespace(), argparse.Namespace(servicer_impl=None)):
            environ = dict(exported)
            launcher.resolve_servicer_impl(args, environ)
            assert environ == exported, (args, exported)


def test_the_launchers_answer_with_the_shared_switch():
    for launcher in (sglang_rust, tokenspeed_rust):
        env = launcher.SERVICER_IMPL_ENV
        assert launcher.servicer_impl_source(environ={}) == ("rust", "default")
        assert launcher.servicer_impl_source(environ={env: "python"}) == ("python", "env")
        flag = argparse.Namespace(servicer_impl="python")
        assert launcher.servicer_impl_source(flag, environ={}) == ("python", "flag")
    assert vllm_rust.servicer_impl_source(environ={}) == ("rust", "default")
    assert vllm_rust.IMPLS is sglang_rust.IMPLS is tokenspeed_rust.IMPLS is rust_lifecycle.IMPLS
    assert rust_lifecycle.DEFAULT_IMPL == "rust"


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


def test_a_launch_the_rust_path_does_not_serve_falls_back_to_python_on_defaults(caplog):
    """The launcher's probe is asked only when the default would select Rust:
    a gap it names selects the Python servicer with the origin ``fallback``,
    one warning names the gap, and the decision is written back like a flag's
    (the engine processes and the Python servicer's guard read it there). A
    probe that finds nothing leaves the default; an explicit selection never
    asks the probe."""
    caplog.set_level(logging.WARNING)
    asked = []

    def unserved():
        asked.append(True)
        return "PD disaggregation (disaggregation_mode='prefill')"

    environ = {}
    assert rust_lifecycle.servicer_switch("X_SERVICER_IMPL", None, environ, unserved=unserved) == (
        "python",
        rust_lifecycle.SOURCE_FALLBACK,
    )
    assert environ == {"X_SERVICER_IMPL": "python"}
    assert len(asked) == 1
    line = next(
        record for record in caplog.records if "does not serve this launch yet" in record.message
    )
    assert line.levelno == logging.WARNING
    assert "PD disaggregation (disaggregation_mode='prefill')" in line.message
    assert (
        "an explicit X_SERVICER_IMPL=rust or --servicer-impl rust takes the Rust path regardless"
        in line.message
    )
    # A served launch: the default stands and nothing is written.
    environ = {}
    assert rust_lifecycle.servicer_switch(
        "X_SERVICER_IMPL", None, environ, unserved=lambda: None
    ) == ("rust", "default")
    assert environ == {}
    # An explicit rust (the variable, or the flag) is never second-guessed,
    # and an explicit python needs no probe: the Rust path refuses what it
    # cannot serve, as before.
    asked.clear()
    assert rust_lifecycle.servicer_switch(
        "X_SERVICER_IMPL", None, {"X_SERVICER_IMPL": "rust"}, unserved=unserved
    ) == ("rust", "env")
    assert rust_lifecycle.servicer_switch(
        "X_SERVICER_IMPL", argparse.Namespace(servicer_impl="rust"), {}, unserved=unserved
    ) == ("rust", "flag")
    assert rust_lifecycle.servicer_switch(
        "X_SERVICER_IMPL", argparse.Namespace(servicer_impl="python"), {}, unserved=unserved
    ) == ("python", "flag")
    assert asked == []


def test_the_announcement_names_the_implementation_and_deprecates_python(caplog):
    caplog.set_level(logging.INFO)
    log = logging.getLogger("test.servicer.announce")
    rust_lifecycle.announce_servicer_impl("rust", "default", log)
    assert "Servicer implementation: rust (source=default)" in caplog.text
    assert rust_lifecycle.DEPRECATION_LINE not in caplog.text
    for origin in ("flag", "env", "fallback"):
        caplog.clear()
        rust_lifecycle.announce_servicer_impl("python", origin, log)
        assert f"Servicer implementation: python (source={origin})" in caplog.text
        deprecation = [r for r in caplog.records if r.message == rust_lifecycle.DEPRECATION_LINE]
        assert len(deprecation) == 1 and deprecation[0].levelno == logging.WARNING


def test_each_launcher_names_what_its_rust_path_does_not_serve():
    """The per-engine probes behind the fallback: SGLang's PD/EPD and a launch
    that asks for multimodal input (the launch line only, never the model's
    own modality), TokenSpeed's PD/EPD (the encode role included); vLLM's is
    the launcher the switch cannot bind, probed at the plugin's parse."""
    assert sglang_rust.unserved_on_rust(SimpleNamespace()) is None
    assert sglang_rust.unserved_on_rust(SimpleNamespace(disaggregation_mode="decode")) == (
        "PD disaggregation (disaggregation_mode='decode')"
    )
    assert sglang_rust.unserved_on_rust(SimpleNamespace(encoder_only=True)) == (
        "EPD disaggregation (language_only / encoder_only)"
    )
    assert sglang_rust.unserved_on_rust(SimpleNamespace(enable_multimodal=True)) == (
        "multimodal input (--enable-multimodal; this path carries no media to the scheduler)"
    )
    assert sglang_rust.unserved_on_rust(SimpleNamespace(enable_multimodal=False)) is None
    assert sglang_rust.unserved_on_rust(SimpleNamespace(enable_multimodal=None)) is None
    assert tokenspeed_rust.unserved_on_rust(SimpleNamespace()) is None
    assert tokenspeed_rust.unserved_on_rust(SimpleNamespace(disaggregation_mode="prefill")) == (
        "PD disaggregation (disaggregation_mode='prefill')"
    )
    assert tokenspeed_rust.unserved_on_rust(SimpleNamespace(disaggregation_mode="encode")) == (
        "EPD disaggregation (the encode role)"
    )
    # TokenSpeed's launcher hands the engine's args beside its own flag namespace.
    environ = {}
    assert tokenspeed_rust.servicer_impl_source(
        argparse.Namespace(servicer_impl=None),
        environ,
        engine_args=SimpleNamespace(disaggregation_mode="decode"),
    ) == ("python", "fallback")
    assert environ == {tokenspeed_rust.SERVICER_IMPL_ENV: "python"}
    assert tokenspeed_rust.servicer_impl_source(
        argparse.Namespace(servicer_impl=None), {}, engine_args=SimpleNamespace()
    ) == ("rust", "default")
