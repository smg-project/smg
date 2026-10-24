"""The servicer's metrics endpoint on the three launchers: one flag pair
(``--metrics-port``, ``--metrics-host``) with one precedence (the flag, else
the engine's variable, else no endpoint), one refusal of a clash with the gRPC
port before the engine starts, and the bind address handed to the Rust server
(the Python servicer serves the same address itself: ``test_metrics.py``)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import types
from types import SimpleNamespace

import pytest
from smg_grpc_servicer import metrics_flag, rust_lifecycle
from smg_grpc_servicer.sglang import rust as sglang_rust
from smg_grpc_servicer.tokenspeed import rust as tokenspeed_rust
from smg_grpc_servicer.vllm import rust as vllm_rust

LAUNCHERS = [
    pytest.param(vllm_rust.VllmLauncher, "SMG_VLLM_SERVICER_METRICS", id="vllm"),
    pytest.param(sglang_rust.SglangLauncher, "SMG_SGLANG_SERVICER_METRICS", id="sglang"),
    pytest.param(
        tokenspeed_rust.TokenSpeedLauncher, "SMG_TOKENSPEED_SERVICER_METRICS", id="tokenspeed"
    ),
]


@pytest.fixture(autouse=True)
def nothing_of_the_metrics_flags_in_the_environment(monkeypatch):
    for name in (
        "SMG_VLLM_SERVICER_METRICS_PORT",
        "SMG_VLLM_SERVICER_METRICS_HOST",
        "SMG_SGLANG_SERVICER_METRICS_PORT",
        "SMG_SGLANG_SERVICER_METRICS_HOST",
        "SMG_TOKENSPEED_SERVICER_METRICS_PORT",
        "SMG_TOKENSPEED_SERVICER_METRICS_HOST",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(("launcher_class", "prefix"), LAUNCHERS)
def test_metrics_port_flag_env_and_default_per_launcher(launcher_class, prefix):
    """The same precedence on every launcher: the flag, else the variable,
    else no endpoint; the host from its flag, its variable, else the gRPC
    host; an IPv6 host bracketed; 0 an ephemeral port."""
    launcher = launcher_class()
    assert launcher.metrics_port_env == f"{prefix}_PORT"
    assert launcher.metrics_host_env == f"{prefix}_HOST"
    grpc = SimpleNamespace(host="10.0.0.7", port=50051)
    assert launcher.metrics_bind_address(grpc, environ={}) is None
    assert launcher.metrics_bind_address(grpc, environ={f"{prefix}_PORT": ""}) is None
    assert (
        launcher.metrics_bind_address(grpc, environ={f"{prefix}_PORT": "29100"}) == "10.0.0.7:29100"
    )
    flag = SimpleNamespace(host="10.0.0.7", port=50051, metrics_port=29101, metrics_host=None)
    assert (
        launcher.metrics_bind_address(flag, environ={f"{prefix}_PORT": "29100"}) == "10.0.0.7:29101"
    )
    # The host: its flag, else its variable, else the gRPC host.
    assert (
        launcher.metrics_bind_address(
            flag, environ={f"{prefix}_PORT": "29100", f"{prefix}_HOST": "127.0.0.1"}
        )
        == "127.0.0.1:29101"
    )
    hosted = SimpleNamespace(host="10.0.0.7", port=50051, metrics_port=29101, metrics_host="::")
    assert (
        launcher.metrics_bind_address(hosted, environ={f"{prefix}_HOST": "127.0.0.1"})
        == "[::]:29101"
    )
    # A launcher without a host binds every interface, as the gRPC listener does.
    assert (
        launcher.metrics_bind_address(SimpleNamespace(port=50051, metrics_port=0), environ={})
        == "0.0.0.0:0"
    )
    # A value the servicer cannot use names its source.
    with pytest.raises(rust_lifecycle.ServicerConfigError, match=f"{prefix}_PORT"):
        launcher.metrics_bind_address(grpc, environ={f"{prefix}_PORT": "many"})
    with pytest.raises(rust_lifecycle.ServicerConfigError, match="--metrics-port"):
        launcher.metrics_bind_address(
            SimpleNamespace(host="h", port=50051, metrics_port=70000), environ={}
        )


@pytest.mark.parametrize(("launcher_class", "prefix"), LAUNCHERS)
def test_metrics_port_equal_to_grpc_port_is_refused_before_the_engine_starts(
    launcher_class, prefix
):
    launcher = launcher_class()
    clash = SimpleNamespace(host="127.0.0.1", port=50051, metrics_port=50051)
    with pytest.raises(rust_lifecycle.ServicerConfigError, match="is the gRPC port"):
        launcher.metrics_bind_address(clash, environ={})
    with pytest.raises(rust_lifecycle.ServicerConfigError, match="is the gRPC port"):
        launcher.metrics_bind_address(
            SimpleNamespace(host="127.0.0.1", port=50051), environ={f"{prefix}_PORT": "50051"}
        )
    # Port 0 never clashes: it is an ephemeral one.
    assert (
        launcher.metrics_bind_address(
            SimpleNamespace(host="127.0.0.1", port=0, metrics_port=0), environ={}
        )
        == "127.0.0.1:0"
    )


def test_the_flags_join_a_parser_once_and_export_their_variables(monkeypatch):
    """The flags' actions export their values under the launcher's variables,
    which is how a launcher whose namespace keeps only the engine's own
    fields (SGLang, TokenSpeed) reads a flag; a parse without them leaves the
    environment alone."""
    monkeypatch.delenv("X_PORT", raising=False)
    monkeypatch.delenv("X_HOST", raising=False)
    parser = argparse.ArgumentParser()
    assert (
        metrics_flag.add_metrics_arguments(
            parser, port_env="X_PORT", host_env="X_HOST", server_class="S"
        )
        is True
    )
    assert (
        metrics_flag.add_metrics_arguments(
            parser, port_env="X_PORT", host_env="X_HOST", server_class="S"
        )
        is False
    )
    parsed = parser.parse_args([])
    assert (parsed.metrics_port, parsed.metrics_host) == (None, None)
    assert "X_PORT" not in os.environ and "X_HOST" not in os.environ
    parsed = parser.parse_args(["--metrics-port", "29100", "--metrics-host", "::"])
    assert (parsed.metrics_port, parsed.metrics_host) == (29100, "::")
    assert (os.environ["X_PORT"], os.environ["X_HOST"]) == ("29100", "::")
    monkeypatch.delenv("X_PORT")
    monkeypatch.delenv("X_HOST")
    with pytest.raises(SystemExit):
        parser.parse_args(["--metrics-port", "many"])
    assert "--metrics-port" in parser.format_help()


def test_the_three_launchers_register_the_same_flags():
    """The vLLM plugin adds the flags to a parser that serves gRPC, the SGLang
    plugin and the TokenSpeed launcher to theirs: the same pair, once."""
    from smg_grpc_servicer.sglang import plugin as sglang_plugin
    from smg_grpc_servicer.vllm import plugin as vllm_plugin

    grpc_parser = argparse.ArgumentParser()
    grpc_parser.add_argument("--grpc", action="store_true")
    for add, parser in (
        (vllm_plugin.add_metrics_arguments, grpc_parser),
        (sglang_plugin.add_metrics_arguments, argparse.ArgumentParser()),
        (tokenspeed_rust.add_metrics_arguments, argparse.ArgumentParser()),
    ):
        assert add(parser) is True
        assert add(parser) is False
        actions = parser._option_string_actions  # noqa: SLF001 — argparse's registry
        assert {"--metrics-port", "--metrics-host"} <= set(actions)
    # A parser of another vLLM command is left alone.
    plain = argparse.ArgumentParser()
    assert vllm_plugin.add_metrics_arguments(plain) is False
    assert "--metrics-port" not in plain._option_string_actions  # noqa: SLF001 — argparse's registry


def _install(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent:
        parent_module = sys.modules.get(parent) or _install(monkeypatch, parent)
        monkeypatch.setattr(parent_module, child, module, raising=False)
    return module


class _Launcher(rust_lifecycle.RustLauncher):
    """The lifecycle with every engine-specific hook stubbed: what `serve`
    hands the Rust server is the only thing under test."""

    engine = "Test"
    started = "engine"
    counted = "engine"
    server_class = "TestGrpcServer"
    ipc_prefix = "test-servicer"
    handshake_port_env = "X_HANDSHAKE_PORT"
    drain_secs_env = "X_DRAIN_SECS"
    startup_timeout_secs_env = "X_STARTUP_TIMEOUT_SECS"
    startup_ceiling_secs_env = "X_STARTUP_CEILING_SECS"
    silence_timeout_secs_env = "X_SILENCE_TIMEOUT_SECS"
    prefill_tokens_per_sec_env = "X_PREFILL_TOKENS_PER_SEC"
    kv_events_env = "X_KV_EVENTS"
    metrics_port_env = "X_METRICS_PORT"
    metrics_host_env = "X_METRICS_HOST"
    without_tokenizer = "nothing"

    def kv_events_default(self, args):
        return None

    def facts(self, args):
        return {"data_parallel_size": 1, "kv_events_endpoint": "", "model_path": "org/m"}

    def tokenizer_dir(self, args):
        return None

    def launch(self, args, handshake_port, facts):
        raise RuntimeError("the launch is where this test stops")


@pytest.mark.parametrize(
    ("args", "environ", "expected"),
    [
        (SimpleNamespace(host="127.0.0.1", port=50051), {}, None),
        (SimpleNamespace(host="127.0.0.1", port=50051), {"X_METRICS_PORT": "0"}, "127.0.0.1:0"),
        (
            SimpleNamespace(host="127.0.0.1", port=50051, metrics_port=29100, metrics_host="::1"),
            {},
            "[::1]:29100",
        ),
    ],
    ids=["off", "env", "flags"],
)
def test_serve_passes_the_metrics_bind_address_to_the_rust_server(
    monkeypatch, tmp_path, args, environ, expected
):
    created: dict = {}
    stopped: list = []

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.address = kwargs["bind_address"]
            self.metrics_address = kwargs["metrics_bind_address"]

        def stop(self, timeout_secs=5.0):
            stopped.append(timeout_secs)

    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    _install(monkeypatch, "smg")
    _install(
        monkeypatch,
        "smg.servicer",
        TestGrpcServer=FakeServer,
        init_servicer_tracing=lambda level=None: None,
    )
    with pytest.raises(RuntimeError, match="the launch is where this test stops"):
        asyncio.run(_Launcher().serve(args))
    assert created["bind_address"] == "127.0.0.1:50051"
    assert created["metrics_bind_address"] == expected
    assert stopped, "a launch that fails stops the already bound server"


def test_serve_refuses_a_metrics_port_equal_to_the_grpc_port_before_the_server_binds(
    monkeypatch, tmp_path
):
    created: list = []

    class FakeServer:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    _install(monkeypatch, "smg")
    _install(
        monkeypatch,
        "smg.servicer",
        TestGrpcServer=FakeServer,
        init_servicer_tracing=lambda level=None: None,
    )
    with pytest.raises(rust_lifecycle.ServicerConfigError, match="is the gRPC port"):
        asyncio.run(
            _Launcher().serve(SimpleNamespace(host="127.0.0.1", port=50051, metrics_port=50051))
        )
    assert created == [], "nothing was bound"
