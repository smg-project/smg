"""Unit tests for the Rust request path of the TensorRT-LLM servicer
(``smg_grpc_servicer.trtllm.rust``).

Everything engine-shaped is lazy or injected: the facts read the arguments
and a checkpoint directory of the test's making, the shim is a fake module
found by import, ``serve_rust`` gets a fake Rust server class and a fake
launch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import types

import pytest
from smg_grpc_servicer import constrained_decoding as cd
from smg_grpc_servicer import rust_lifecycle
from smg_grpc_servicer.trtllm import __main__ as cli
from smg_grpc_servicer.trtllm import rust
from smg_grpc_servicer.trtllm.rust import ServeArgs


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


def _install_fake_servicer(monkeypatch, server):
    _install(monkeypatch, "smg")
    return _install(
        monkeypatch,
        "smg.servicer",
        TrtllmGrpcServer=server,
        init_servicer_tracing=lambda *_args, **_kwargs: None,
    )


def _checkpoint(tmp_path, **config) -> str:
    directory = tmp_path / "ckpt"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


def test_package_exposes_the_flag_and_the_rust_entry_points():
    from smg_grpc_servicer import trtllm

    assert trtllm.SERVICER_IMPL_ENV == "SMG_TRTLLM_SERVICER_IMPL"
    assert trtllm.resolve_servicer_impl is rust.resolve_servicer_impl
    assert trtllm.serve_rust is rust.serve_rust
    assert trtllm.serve_rust_from_engine is rust.serve_rust_from_engine


def test_the_switch_reads_the_flag_then_the_variable_then_python():
    env = rust.SERVICER_IMPL_ENV
    assert rust.resolve_servicer_impl(environ={}) == "python"
    assert rust.resolve_servicer_impl(environ={env: " Rust "}) == "rust"
    flagged = argparse.Namespace(servicer_impl="rust")
    assert rust.resolve_servicer_impl(flagged, environ={env: "python"}) == "rust"
    assert rust.servicer_impl_source(environ={env: "rust"}) == ("rust", "env")
    with pytest.raises(rust_lifecycle.ServicerConfigError):
        rust.resolve_servicer_impl(environ={env: "go"})


def test_model_facts_read_the_engine_arguments_and_the_checkpoint(tmp_path):
    model = _checkpoint(
        tmp_path,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        eos_token_id=[2, 3],
    )
    args = ServeArgs(
        model=model,
        served_model_name="served",
        llm_args={"model": model, "max_seq_len": 8192, "max_input_len": 4096, "max_batch_size": 64},
    )
    facts = rust.model_facts(args)
    assert facts == {
        "model_path": model,
        "served_model_name": "served",
        "max_input_len": 4096,
        "max_seq_len": 8192,
        "max_batch_size": 64,
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "eos_token_ids": [2, 3],
    }
    # A nested text config (a VLM checkpoint) and a scalar eos id.
    nested = (
        _checkpoint(
            tmp_path / "vlm",
            text_config={"vocab_size": 1000, "num_hidden_layers": 4, "eos_token_id": 7},
        )
        if (tmp_path / "vlm").mkdir() is None
        else None
    )
    facts = rust.model_facts(ServeArgs(model=nested, llm_args={}))
    assert facts["vocab_size"] == 1000 and facts["num_layers"] == 4
    assert facts["eos_token_ids"] == [7]
    assert facts["max_seq_len"] == 0, "unstated: the handshake fills it"
    assert facts["served_model_name"] == nested


def test_model_facts_without_a_local_checkpoint_leave_the_config_facts_at_zero(monkeypatch):
    monkeypatch.setattr(rust, "resolve_tokenizer_dir", lambda *_: None)
    facts = rust.model_facts(ServeArgs(model="org/model", llm_args={"max_seq_len": 10}))
    assert facts["vocab_size"] == 0 and facts["eos_token_ids"] == []
    assert facts["max_seq_len"] == 10


def test_server_facts_name_the_backend_the_sizes_and_the_grammar_backend(monkeypatch):
    monkeypatch.setattr(rust, "_engine_version", lambda: "1.3.0")
    monkeypatch.setattr(cd, "installed_xgrammar_version", lambda: "0.2.7")
    facts = rust.server_facts(
        ServeArgs(
            model="m",
            llm_args={
                "backend": "pytorch",
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "guided_decoding_backend": "XGrammar",
                "max_batch_size": 32,
            },
        )
    )
    assert facts == {
        "backend": "pytorch",
        "trtllm_version": "1.3.0",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "data_parallel_size": 1,
        "guided_decoding_backend": "xgrammar",
        "structural_tag_vocabulary": "",
        "max_running_requests": 32,
    }
    # No guided-decoding backend: the engine enforces no grammar.
    bare = rust.server_facts(ServeArgs(model="m", llm_args={}))
    assert bare["guided_decoding_backend"] == "none"
    assert bare["backend"] == "pytorch" and bare["tensor_parallel_size"] == 1


def test_locate_shim_prefers_the_in_tree_package_then_the_overlay(monkeypatch):
    with pytest.raises(rust_lifecycle.ServicerConfigError) as refused:
        rust.locate_shim(("no_such_shim_a", "no_such_shim_b"))
    assert "no_such_shim_a" in str(refused.value) and "no_such_shim_b" in str(refused.value)
    _install(monkeypatch, "trtllm_smg_shim", run=lambda **_: 0)
    assert rust.locate_shim(("no_such_shim_a", "trtllm_smg_shim")) == "trtllm_smg_shim"
    _install(monkeypatch, "tensorrt_llm.grpc.smg.shim", run=lambda **_: 0)
    assert rust.locate_shim() == "tensorrt_llm.grpc.smg.shim"


def test_locate_shim_reads_the_tree_without_importing_the_engine(monkeypatch, tmp_path):
    """A shim found on disk is named without importing its package: an engine
    package that loads its runtime on import must not be imported into the
    servicer's process (the shim child imports it)."""
    package = tmp_path / "fakeengine_w2"
    (package / "grpc" / "smg" / "shim").mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('the engine was imported')\n")
    (package / "grpc" / "__init__.py").write_text("")
    (package / "grpc" / "smg" / "shim" / "__init__.py").write_text(
        "def run(**kwargs):\n    return 0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fakeengine_w2", raising=False)
    assert (
        rust.locate_shim(("fakeengine_w2.grpc.smg.shim", "trtllm_smg_shim"))
        == "fakeengine_w2.grpc.smg.shim"
    )
    assert "fakeengine_w2" not in sys.modules
    with pytest.raises(rust_lifecycle.ServicerConfigError) as refused:
        rust.locate_shim(("fakeengine_w2.grpc.smg.other",))
    assert "is not under 'fakeengine_w2'" in str(refused.value)


def test_launch_shim_spawns_the_entry_against_the_handshake(monkeypatch):
    started = {}

    class FakeProcess:
        pid = 4242
        exitcode = None

        def __init__(self, *, target, args, name):
            started.update(target=target, args=args, name=name)

        def start(self):
            started["started"] = True

    class FakeContext:
        Process = FakeProcess

    monkeypatch.setattr(rust.multiprocessing, "get_context", lambda method: FakeContext())
    args = ServeArgs(
        model="m", llm_args={"model": "m", "tensor_parallel_size": 2}, served_model_name="s"
    )
    engine = rust.launch_shim(args, handshake_port=24321, module="trtllm_smg_shim")
    assert isinstance(engine, rust_lifecycle.EngineProcess) and engine.pid == 4242
    assert started["target"] is rust._run_shim
    assert started["args"] == (
        "trtllm_smg_shim",
        "tcp://127.0.0.1:24321",
        0,
        {"model": "m", "tensor_parallel_size": 2},
        "s",
    )
    assert started["name"] == "smg-trtllm-shim" and started["started"]


def test_the_child_target_runs_the_shims_entry(monkeypatch):
    calls = {}
    _install(monkeypatch, "trtllm_smg_shim", run=lambda **kwargs: calls.update(kwargs) or 3)
    with pytest.raises(SystemExit) as exited:
        rust._run_shim("trtllm_smg_shim", "tcp://127.0.0.1:1", 0, {"model": "m"}, "s")
    assert exited.value.code == 3
    assert calls == {
        "handshake_address": "tcp://127.0.0.1:1",
        "engine_index": 0,
        "llm_args": {"model": "m"},
        "served_model_name": "s",
    }


def test_serve_rust_wires_the_server_the_shim_and_the_supervisor(monkeypatch, tmp_path):
    created = {}

    class FakeServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.address = kwargs["bind_address"]

    _install_fake_servicer(monkeypatch, FakeServer)
    _install(monkeypatch, "trtllm_smg_shim", run=lambda **_: 0)
    # The shim this test installs is the one the launcher must find, on a host
    # with the engine tree (and its own shim package) as well as without.
    monkeypatch.setattr(rust, "SHIM_MODULES", ("trtllm_smg_shim",))
    monkeypatch.setenv(rust.HANDSHAKE_PORT_ENV, "24321")
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    monkeypatch.setattr(rust, "model_facts", lambda args: {"model_path": args.model})
    monkeypatch.setattr(
        rust, "server_facts", lambda args: {"data_parallel_size": 1, "trtllm_version": "x"}
    )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: "/tok")
    launched = {}

    def fake_launch(args, *, handshake_port, module=None):
        launched.update(args=args, port=handshake_port, module=module)
        return "engine"

    monkeypatch.setattr(rust, "launch_shim", fake_launch)

    async def fake_supervise(server, engine, *, drain_secs):
        assert isinstance(server, FakeServer) and engine == "engine"
        return 7

    monkeypatch.setattr(rust_lifecycle, "supervise", fake_supervise)
    args = ServeArgs(
        model="org/model", host="127.0.0.1", port=30011, llm_args={"model": "org/model"}
    )
    assert asyncio.run(rust.serve_rust(args)) == 7
    assert created["bind_address"] == "127.0.0.1:30011"
    assert created["handshake_address"] == "tcp://127.0.0.1:24321"
    assert created["ipc_base_url"].startswith(f"ipc://{tmp_path}/trtllm-servicer-")
    assert created["engine_count"] == 1 and created["tokenizer_dir"] == "/tok"
    assert created["model_path"] == "org/model" and created["trtllm_version"] == "x"
    assert launched == {"args": args, "port": 24321, "module": "trtllm_smg_shim"}


def test_serve_rust_refuses_a_missing_shim_before_the_server_binds(monkeypatch, tmp_path):
    class FakeServer:
        def __init__(self, **kwargs):
            pytest.fail("the server must not bind without the shim")

    _install_fake_servicer(monkeypatch, FakeServer)
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    monkeypatch.setattr(rust, "SHIM_MODULES", ("no_such_shim_a", "no_such_shim_b"))
    with pytest.raises(rust_lifecycle.ServicerConfigError, match="no_such_shim_a"):
        asyncio.run(rust.serve_rust(ServeArgs(model="m", llm_args={"model": "m"})))


def test_serve_rust_from_engine_builds_the_arguments_the_engine_hands_over(monkeypatch):
    seen = {}

    async def fake_serve_rust(args):
        seen["args"] = args
        return 5

    monkeypatch.setattr(rust, "serve_rust", fake_serve_rust)
    code = rust.serve_rust_from_engine(
        "0.0.0.0", 50051, {"model": "org/model", "backend": "pytorch"}, served_model_name="served"
    )
    assert code == 5
    args = seen["args"]
    assert args.model == "org/model" and args.host == "0.0.0.0" and args.port == 50051
    assert args.llm_args == {"model": "org/model", "backend": "pytorch"}
    assert args.served_model_name == "served"


def test_the_cli_hands_the_process_to_rust_when_the_flag_says_so(monkeypatch, tmp_path):
    seen = {}

    async def fake_serve_rust(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "serve_rust", fake_serve_rust)
    # The CLI's actions write the process environment; registering the
    # variable first lets the fixture restore it after the test.
    monkeypatch.setenv(rust.SERVICER_IMPL_ENV, "python")
    options = tmp_path / "llm.json"
    options.write_text(json.dumps({"tensor_parallel_size": 2}), encoding="utf-8")
    with pytest.raises(SystemExit) as exited:
        cli.main(
            [
                "--model",
                "org/model",
                "--port",
                "50051",
                "--llm-args",
                f"@{options}",
                "--servicer-impl",
                "rust",
            ]
        )
    assert exited.value.code == 0
    args = seen["args"]
    assert args.llm_args == {"tensor_parallel_size": 2, "model": "org/model", "backend": "pytorch"}
    assert args.port == 50051 and args.servicer_impl == "rust"


def test_the_cli_runs_the_engines_own_servicer_by_default(monkeypatch):
    launched = {}
    _install(
        monkeypatch,
        "tensorrt_llm.grpc.smg.server",
        launch_smg_server=lambda host, port, llm_args, served_model_name=None: launched.update(
            host=host, port=port, llm_args=llm_args, served_model_name=served_model_name
        ),
    )
    monkeypatch.delenv(rust.SERVICER_IMPL_ENV, raising=False)
    cli.main(
        ["--model", "org/model", "--host", "127.0.0.1", "--port", "1", "--served-model-name", "s"]
    )
    assert launched == {
        "host": "127.0.0.1",
        "port": 1,
        "llm_args": {"model": "org/model", "backend": "pytorch"},
        "served_model_name": "s",
    }


def test_a_metrics_port_under_the_python_servicer_is_refused(monkeypatch):
    monkeypatch.delenv(rust.SERVICER_IMPL_ENV, raising=False)
    monkeypatch.setenv(rust.METRICS_PORT_ENV, "0")
    with pytest.raises(SystemExit) as exited:
        cli.main(["--model", "org/model", "--metrics-port", "9000"])
    assert exited.value.code == 2


def test_server_facts_carry_the_structural_tag_vocabulary_of_an_older_xgrammar(monkeypatch):
    # The xgrammar installed beside the engine predates the structural-tag
    # nodes the Router writes (0.1.33 is the first release with them): the
    # fact says so under `guided_decoding_backend: xgrammar`; no backend (the
    # engine enforces no grammar), another backend, or a current xgrammar
    # says nothing.
    monkeypatch.setattr(rust, "_engine_version", lambda: "1.3.0")
    monkeypatch.setattr(cd, "installed_xgrammar_version", lambda: "0.1.32")

    def facts(**llm_args):
        return rust.server_facts(ServeArgs(model="m", llm_args=llm_args))

    assert facts(guided_decoding_backend="xgrammar")["structural_tag_vocabulary"] == "older"
    assert facts(guided_decoding_backend="XGrammar")["structural_tag_vocabulary"] == "older"
    assert facts()["structural_tag_vocabulary"] == ""
    assert facts(guided_decoding_backend="llguidance")["structural_tag_vocabulary"] == ""
    monkeypatch.setattr(cd, "installed_xgrammar_version", lambda: "0.1.33")
    assert facts(guided_decoding_backend="xgrammar")["structural_tag_vocabulary"] == ""
