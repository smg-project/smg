"""Unit tests for the Rust request path of the vLLM servicer
(``smg_grpc_servicer.vllm.rust``).

Everything vLLM-shaped is lazy in the module, so these run without vLLM:
``serve_rust`` gets stub ``vllm`` modules, a fake Rust server class and a fake
engine launch; the lifecycle loop gets fake handles.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.vllm import rust


def _install(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent:
        parent_module = sys.modules.get(parent) or _install(monkeypatch, parent)
        setattr(parent_module, child, module)
    return module


def test_package_exposes_the_flag_and_the_rust_entry_points():
    import smg_grpc_servicer.vllm as pkg

    assert pkg.SERVICER_IMPL_ENV == "SMG_VLLM_SERVICER_IMPL"
    assert pkg.resolve_servicer_impl is rust.resolve_servicer_impl
    assert pkg.serve_rust is rust.serve_rust


def test_resolve_servicer_impl_prefers_the_launcher_flag_then_the_env():
    assert rust.resolve_servicer_impl(environ={}) == "python"
    assert rust.resolve_servicer_impl(environ={"SMG_VLLM_SERVICER_IMPL": " Rust "}) == "rust"
    args = argparse.Namespace(servicer_impl="rust")
    assert rust.resolve_servicer_impl(args, environ={}) == "rust"
    # An unset launcher flag falls through to the env.
    args = argparse.Namespace(servicer_impl=None)
    assert rust.resolve_servicer_impl(args, environ={"SMG_VLLM_SERVICER_IMPL": "rust"}) == "rust"
    with pytest.raises(ValueError, match="SMG_VLLM_SERVICER_IMPL"):
        rust.resolve_servicer_impl(environ={"SMG_VLLM_SERVICER_IMPL": "go"})


def test_a_launcher_flag_decision_is_written_back_for_the_python_guard():
    # `--servicer-impl python` with `rust` still exported: the hook picks
    # Python, and the servicer's guard must see that decision, not the env.
    environ = {"SMG_VLLM_SERVICER_IMPL": "rust"}
    args = argparse.Namespace(servicer_impl="python")
    assert rust.resolve_servicer_impl(args, environ=environ) == "python"
    assert environ["SMG_VLLM_SERVICER_IMPL"] == "python"
    rust.require_python_impl(environ=environ)
    # Without a flag in hand nothing is written.
    environ = {}
    assert rust.resolve_servicer_impl(environ=environ) == "python"
    assert environ == {}


def test_python_servicer_refuses_to_start_when_the_flag_asks_for_rust():
    rust.require_python_impl(environ={})
    rust.require_python_impl(environ={"SMG_VLLM_SERVICER_IMPL": "python"})
    with pytest.raises(RuntimeError, match="does not consult"):
        rust.require_python_impl(environ={"SMG_VLLM_SERVICER_IMPL": "rust"})


def test_upstream_hook_is_detected_in_the_launcher_source(tmp_path):
    launcher = tmp_path / "vllm" / "entrypoints" / "launchers"
    launcher.mkdir(parents=True)
    launcher.joinpath("grpc_server.py").write_text("async def serve_grpc(args):\n    pass\n")
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is False
    launcher.joinpath("grpc_server.py").write_text(
        "from smg_grpc_servicer.vllm import resolve_servicer_impl, serve_rust\n"
    )
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is True
    # A moved launcher is found anywhere under entrypoints/.
    launcher.joinpath("grpc_server.py").unlink()
    (tmp_path / "vllm" / "entrypoints" / "serve_grpc.py").write_text("resolve_servicer_impl\n")
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is True
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "nowhere")) is False


def test_require_upstream_hook_names_the_fix(monkeypatch):
    monkeypatch.setattr(rust, "upstream_hook_installed", lambda: False)
    with pytest.raises(RuntimeError, match="resolve_servicer_impl"):
        rust.require_upstream_hook()
    monkeypatch.setattr(rust, "upstream_hook_installed", lambda: True)
    rust.require_upstream_hook()


# ---------------------------------------------------------------------------
# Model / server info
# ---------------------------------------------------------------------------


def _config(**overrides):
    model_config = SimpleNamespace(
        model="org/m",
        served_model_name=["served-a", "served-b"],
        tokenizer="org/m-tok",
        runner_type="generate",
        is_multimodal_model=False,
        max_model_len=4096,
        get_vocab_size=lambda: 1024,
        hf_config=SimpleNamespace(
            model_type="qwen3", eos_token_id=[151645, 151643], pad_token_id=None, bos_token_id=1
        ),
        architectures=["Qwen3ForCausalLM"],
        try_get_generation_config=lambda: {"eos_token_id": [151643, 7], "temperature": 0.6},
        get_diff_sampling_param=lambda: {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": None,
            "max_new_tokens": 10,
        },
    )
    for key, value in overrides.items():
        setattr(model_config, key, value)
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(data_parallel_size=2),
    )


def test_model_info_mirrors_the_python_servicer(monkeypatch):
    monkeypatch.setenv("SMG_PAIRING_PROTOCOL", " nixl ")
    info = rust.model_info_from_config(_config())
    assert info["model_path"] == "org/m"
    assert info["served_model_name"] == "served-a"
    assert info["tokenizer_path"] == "org/m-tok"
    assert info["is_generation"] is True
    assert info["max_context_length"] == 4096
    assert info["vocab_size"] == 1024
    # A text model: vLLM's own multimodal check says no.
    assert info["supports_vision"] is False
    assert info["model_type"] == "qwen3"
    assert info["architectures"] == ["Qwen3ForCausalLM"]
    # Model config ids first (the primary EOS), generation-config extras after, deduped.
    assert info["eos_token_ids"] == [151645, 151643, 7]
    assert info["pad_token_id"] == 0
    assert info["bos_token_id"] == 1
    # None values and non-sampling keys are dropped, as the Python servicer does.
    assert json.loads(info["default_sampling_params_json"]) == {"temperature": 0.6, "top_p": 0.95}
    assert info["data_parallel_size"] == 2
    assert info["pairing_protocol"] == "nixl"
    # No KV connector configured: the PD identity is empty, as in Python.
    assert (info["kv_connector"], info["kv_role"], info["kv_engine_id"]) == ("", "", "")
    assert info["block_size"] == 0 and info["model_dtype"] == ""
    # No KV-event publisher configured, backend unset, and the host's shm id.
    assert info["kv_events_endpoint"] == "" and info["kv_events_topic"] == ""
    assert info["structured_outputs_backend"] == "auto"
    assert isinstance(info["shm_namespace_id"], str)
    # No pooler config on a generation model.
    assert info["pooler_use_activation"] is None and info["pooler_dimensions"] is None


def test_model_info_carries_the_pooler_config():
    config = _config(
        runner_type="pooling",
        pooler_config=SimpleNamespace(use_activation=False, dimensions=256, pooling_type="MEAN"),
    )
    info = rust.model_info_from_config(config)
    assert info["is_generation"] is False
    assert info["pooler_use_activation"] is False
    assert info["pooler_dimensions"] == 256
    config = _config(pooler_config=SimpleNamespace(use_activation=None, dimensions=None))
    info = rust.model_info_from_config(config)
    assert info["pooler_use_activation"] is None and info["pooler_dimensions"] is None


def test_model_info_reports_the_pd_identity_and_pairing_facts():
    config = _config(dtype="torch.bfloat16")
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="NixlConnector", kv_role="kv_producer", engine_id="eng-a"
    )
    config.cache_config = SimpleNamespace(cache_dtype="auto", block_size=16)
    config.attention_config = SimpleNamespace(backend=SimpleNamespace(name="FLASH_ATTN"))
    info = rust.model_info_from_config(config)
    assert info["kv_connector"] == "NixlConnector"
    assert info["kv_role"] == "kv_producer"
    assert info["kv_engine_id"] == "eng-a"
    # The same labels the Python servicer's GetServerInfo derives.
    assert info["kv_cache_dtype"] == "auto"
    assert info["block_size"] == 16
    assert info["attention_backend"] == "FLASH_ATTN"
    assert info["model_dtype"] == "torch.bfloat16"


def test_model_info_reports_vision_from_vllms_own_check():
    config = _config(supports_multimodal_inputs=True, is_multimodal_model=True)
    assert rust.model_info_from_config(config)["supports_vision"] is True
    # `--language-model-only` drops the encoder: vLLM says no, so do we.
    config = _config(supports_multimodal_inputs=False, is_multimodal_model=True)
    assert rust.model_info_from_config(config)["supports_vision"] is False


def test_model_info_reports_kv_events_and_the_structured_backend():
    config = _config()
    config.structured_outputs_config = SimpleNamespace(backend="xgrammar")
    config.kv_events_config = SimpleNamespace(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="tcp://*:5557",
        replay_endpoint="tcp://*:5558",
        topic="kv",
    )
    info = rust.model_info_from_config(config)
    assert info["structured_outputs_backend"] == "xgrammar"
    assert info["kv_events_endpoint"] == "tcp://*:5557"
    assert info["kv_events_replay_endpoint"] == "tcp://*:5558"
    assert info["kv_events_topic"] == "kv"
    # A non-ZMQ publisher (or disabled events) leaves the relay off.
    config.kv_events_config.publisher = "null"
    assert rust.model_info_from_config(config)["kv_events_endpoint"] == ""
    config.kv_events_config.publisher = "zmq"
    config.kv_events_config.enable_kv_cache_events = False
    assert rust.model_info_from_config(config)["kv_events_endpoint"] == ""


def test_model_info_with_scalar_eos_and_no_generation_config():
    config = _config(
        hf_config=SimpleNamespace(
            model_type="llama", eos_token_id=2, pad_token_id=0, bos_token_id=1
        ),
        try_get_generation_config=lambda: None,
        get_diff_sampling_param=lambda: {},
        served_model_name=None,
    )
    info = rust.model_info_from_config(config)
    assert info["eos_token_ids"] == [2]
    assert info["served_model_name"] == "org/m"
    assert info["default_sampling_params_json"] == ""


# ---------------------------------------------------------------------------
# Headless engine launch
# ---------------------------------------------------------------------------


def _stub_frontend_args(monkeypatch):
    @dataclasses.dataclass
    class FrontendArgs:
        host: str | None = None
        port: int = 8000
        reasoning_parser_plugin: str = ""
        middleware: list = dataclasses.field(default_factory=list)

    _install(monkeypatch, "vllm")
    _install(monkeypatch, "vllm.entrypoints")
    _install(monkeypatch, "vllm.entrypoints.launchers")
    _install(monkeypatch, "vllm.entrypoints.launchers.cli_args", FrontendArgs=FrontendArgs)


def test_headless_namespace_is_built_from_the_parsed_args_not_argv(monkeypatch):
    """The `python -m vllm.entrypoints.grpc_server` namespace: engine args
    stay as parsed, the frontend fields it lacks get defaults, and the
    data-parallel group is pinned to this host and the handshake port."""
    _stub_frontend_args(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "org/m", "--grpc", "--port", "50051"])
    args = argparse.Namespace(host="127.0.0.1", port=50051, model="org/m", max_model_len=4096)
    ns = rust.headless_namespace(args, handshake_port=24321, data_parallel_size=2)
    assert ns.model == "org/m" and ns.model_tag is None
    assert ns.max_model_len == 4096
    assert ns.headless is True and ns.grpc is False and ns.api_server_count == 0
    assert ns.data_parallel_size == 2 and ns.data_parallel_size_local == 2
    assert ns.data_parallel_address == "127.0.0.1"
    assert ns.data_parallel_rpc_port == 24321
    assert ns.data_parallel_start_rank is None
    assert ns.data_parallel_hybrid_lb is False and ns.data_parallel_external_lb is False
    # Frontend defaults filled for `run_headless`; parsed values untouched.
    assert ns.reasoning_parser_plugin == "" and ns.middleware == []
    assert ns.host == "127.0.0.1" and ns.port == 50051
    # The launcher's own namespace is not mutated.
    assert not hasattr(args, "headless")


def test_headless_namespace_from_the_vllm_serve_grpc_form(monkeypatch):
    """`vllm serve org/m --grpc`: the positional is already on `args.model`;
    `--grpc` is cleared so the child never re-enters the gRPC path, and every
    frontend field is already present."""
    _stub_frontend_args(monkeypatch)
    args = argparse.Namespace(
        model_tag="org/m",
        model="org/m",
        grpc=True,
        headless=False,
        api_server_count=None,
        host=None,
        port=8000,
        reasoning_parser_plugin="",
        middleware=[],
        tensor_parallel_size=2,
    )
    ns = rust.headless_namespace(args, handshake_port=24321, data_parallel_size=1)
    assert ns.grpc is False and ns.headless is True and ns.model_tag is None
    assert ns.model == "org/m" and ns.tensor_parallel_size == 2
    assert ns.api_server_count == 0
    assert ns.data_parallel_rpc_port == 24321 and ns.data_parallel_size_local == 1


def test_engine_process_is_popen_shaped():
    class Proc:
        pid = 4242
        exitcode = None
        events: list[str] = []

        def terminate(self):
            self.events.append("terminate")

        def kill(self):
            self.events.append("kill")

        def join(self, timeout=None):
            self.events.append(f"join:{timeout}")

    proc = Proc()
    engine = rust.EngineProcess(proc)
    assert engine.pid == 4242 and engine.poll() is None
    with pytest.raises(subprocess.TimeoutExpired):
        engine.wait(timeout=0.1)
    proc.exitcode = 0
    assert engine.wait() == 0 and engine.poll() == 0
    engine.terminate()
    engine.kill()
    assert proc.events == ["join:0.1", "join:None", "terminate", "kill"]


def test_launch_headless_engine_spawns_a_fresh_interpreter(monkeypatch):
    recorded: dict = {}

    class Process:
        def __init__(self, target, args, name):
            recorded["target"] = target
            recorded["args"] = args
            recorded["name"] = name
            self.pid = 7
            self.exitcode = None

        def start(self):
            recorded["started"] = True

    monkeypatch.setattr(
        rust.multiprocessing,
        "get_context",
        lambda method: recorded.setdefault("method", method) and SimpleNamespace(Process=Process),
    )
    ns = argparse.Namespace(model="org/m")
    engine = rust.launch_headless_engine(ns)
    assert recorded["method"] == "spawn"
    assert recorded["target"] is rust._run_headless
    assert recorded["args"] == (ns,)
    assert recorded["started"] is True
    assert engine.pid == 7


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class FakeServer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.address = "127.0.0.1:50051"
        self.engine_ready = False
        self.last_error = None
        self.running = True
        self.events: list[str] = []

    def set_serving(self, serving: bool) -> None:
        self.events.append(f"serving:{serving}")

    def stop(self, timeout: float) -> None:
        self.events.append(f"stop:{timeout}")


class FakeEngine:
    def __init__(self, *_args, **_kwargs):
        self.returncode = None
        self.pid = 99
        self.events: list[str] = []

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self) -> None:
        self.events.append("kill")


def _run_supervise(server, engine, *, before, drain_secs=0.0):
    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(
            rust.supervise(
                server,
                engine,
                drain_secs=drain_secs,
                stop_timeout=1.0,
                stop_event=stop,
                poll_secs=0.01,
            )
        )
        await asyncio.sleep(0.05)
        before(stop)
        return await task

    return asyncio.run(run())


def test_supervise_exits_nonzero_when_the_engine_dies():
    server, engine = FakeServer(), FakeEngine()

    def engine_dies(_stop):
        engine.returncode = 3

    assert _run_supervise(server, engine, before=engine_dies) == 1
    assert server.events == ["serving:False", "stop:1.0"]
    assert engine.events == []


def test_supervise_exits_nonzero_on_a_server_error_and_terminates_the_engine():
    server, engine = FakeServer(), FakeEngine()

    def server_fails(_stop):
        server.last_error = "engine connection failed: handshake timed out"

    assert _run_supervise(server, engine, before=server_fails) == 1
    assert server.events == ["serving:False", "stop:1.0"]
    assert engine.events == ["terminate"]


def test_supervise_drains_then_stops_on_a_signal():
    server, engine = FakeServer(), FakeEngine()
    server.engine_ready = True
    assert _run_supervise(server, engine, before=lambda stop: stop.set(), drain_secs=0.01) == 0
    # Draining is announced before the stop, with the engine still up for it.
    assert server.events == ["serving:False", "stop:1.0"]
    assert engine.events == ["terminate"]


def test_configure_logging_gives_the_package_a_handler(monkeypatch):
    import logging

    root = logging.getLogger()
    pkg = logging.getLogger("smg_grpc_servicer")
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(pkg, "handlers", [])
    monkeypatch.setattr(logging.getLogger("vllm"), "handlers", [])
    rust.configure_logging()
    assert root.handlers or pkg.handlers


def test_serve_rust_wires_the_server_the_engine_and_the_supervisor(monkeypatch, tmp_path):
    """The upstream hook hands `serve_rust` its parsed namespace: the Rust
    server binds the launcher's host/port, the headless engine is launched
    from that same namespace (plus the handshake), and both are supervised.
    argv is never consulted, so the `vllm serve <model> --grpc` form works."""
    recorded: dict = {}

    class AsyncEngineArgs:
        @staticmethod
        def from_cli_args(args):
            recorded["engine_args_from"] = args
            return SimpleNamespace(create_engine_config=lambda usage_context: _config())

    _stub_frontend_args(monkeypatch)
    _install(monkeypatch, "vllm.engine.arg_utils", AsyncEngineArgs=AsyncEngineArgs)
    _install(monkeypatch, "vllm.usage.usage_lib", UsageContext=SimpleNamespace(OPENAI_API_SERVER=1))
    _install(monkeypatch, "smg")
    _install(
        monkeypatch,
        "smg.servicer",
        VllmGrpcServer=FakeServer,
        init_servicer_tracing=lambda: recorded.setdefault("tracing", True),
    )
    launched: list[argparse.Namespace] = []

    def fake_launch(ns):
        launched.append(ns)
        return FakeEngine()

    monkeypatch.setattr(rust, "launch_headless_engine", fake_launch)
    monkeypatch.setattr(rust, "resolve_tokenizer_dir", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(rust, "free_port", lambda: 24321)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "org/m", "--grpc", "--port", "50051"])
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("SMG_VLLM_SERVICER_DRAIN_SECS", "0")

    async def fake_supervise(server, engine, *, drain_secs, **_kwargs):
        recorded["supervised"] = (server, engine, drain_secs)
        return 0

    monkeypatch.setattr(rust, "supervise", fake_supervise)

    args = argparse.Namespace(
        model_tag="org/m", model="org/m", grpc=True, host=None, port=50051, max_model_len=4096
    )
    assert asyncio.run(rust.serve_rust(args)) == 0

    server, engine, drain_secs = recorded["supervised"]
    assert isinstance(server, FakeServer) and isinstance(engine, FakeEngine)
    assert drain_secs == 0.0
    assert recorded["tracing"] is True

    # A launch that fails stops the already-bound server before propagating.
    servers: list[FakeServer] = []

    def recording_server(**kwargs):
        servers.append(FakeServer(**kwargs))
        return servers[-1]

    _install(
        monkeypatch,
        "smg.servicer",
        VllmGrpcServer=recording_server,
        init_servicer_tracing=lambda: None,
    )

    def failing_launch(ns):
        raise RuntimeError("engine would not start")

    monkeypatch.setattr(rust, "launch_headless_engine", failing_launch)
    with pytest.raises(RuntimeError, match="engine would not start"):
        asyncio.run(rust.serve_rust(args))
    assert servers[0].events == ["serving:False", "stop:5.0"]
    assert recorded["engine_args_from"] is args
    kwargs = server.kwargs
    # `vllm serve` leaves host unset; upstream binds every interface then.
    assert kwargs["bind_address"] == "0.0.0.0:50051"
    assert kwargs["ipc_base_url"] == f"ipc://{tmp_path}/servicer-{os.getpid()}"
    assert kwargs["handshake_address"] == "tcp://127.0.0.1:24321"
    assert kwargs["engine_count"] == 2
    assert kwargs["tokenizer_dir"] == str(tmp_path)
    assert kwargs["served_model_name"] == "served-a"
    assert kwargs["eos_token_ids"] == [151645, 151643, 7]
    assert kwargs["kv_connector"] == ""

    (ns,) = launched
    assert ns.model == "org/m" and ns.model_tag is None and ns.grpc is False
    assert ns.headless is True and ns.max_model_len == 4096
    assert ns.data_parallel_rpc_port == 24321
    assert ns.data_parallel_size == 2 and ns.data_parallel_size_local == 2
