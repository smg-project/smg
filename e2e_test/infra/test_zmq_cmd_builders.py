"""Unit tests for the ZMQ direct-backend command builders (no GPU)."""

from __future__ import annotations

import pytest
from infra.constants import ConnectionMode
from infra.worker import Worker

_VLLM_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_TS_MODEL = "Qwen/Qwen3.5-9B"


@pytest.fixture
def serve():
    # The ZMQ builders delegate to smg.serve, so the wheel must be importable.
    # Scoped to a fixture (not module import) so the gRPC test below still runs
    # when the wheel is absent.
    return pytest.importorskip("smg.serve")


def _worker(engine, port=50111):
    return Worker(
        model_id=_TS_MODEL if engine == "tokenspeed" else _VLLM_MODEL,
        engine=engine,
        port=port,
        gpu_ids=[0],
        mode=ConnectionMode.ZMQ,
    )


def test_zmq_base_url_matches_serve_helper(serve):
    w = _worker("vllm", port=50123)
    assert w.base_url == serve._zmq_ipc_url(50123)
    assert w.base_url.startswith("ipc://")


def test_vllm_zmq_cmd_is_headless_with_derived_handshake_port(serve):
    w = _worker("vllm")
    cmd = w._build_vllm_zmq_cmd("/models/llama", 1, {"vllm_args": ["--max-model-len", "2048"]})
    assert "serve" in cmd
    assert "--headless" in cmd
    assert "/models/llama" in cmd
    # The engine dials the same tcp port SMG derives from the ipc path.
    expected_port = serve._zmq_handshake_port(serve._zmq_ipc_url(w.port))
    assert cmd[cmd.index("--data-parallel-rpc-port") + 1] == str(expected_port)
    # Model-spec engine args ride through.
    assert cmd[cmd.index("--max-model-len") + 1] == "2048"


def test_sglang_zmq_cmd_runs_the_headless_launcher_with_derived_handshake_port(serve):
    w = _worker("sglang")
    cmd = w._build_sglang_zmq_cmd(
        "/models/qwen", 1, {"sglang_args": ["--mem-fraction-static", "0.5"]}
    )
    # SGLang itself is unchanged: the scheduler is launched headless by the
    # smg servicer package, whose SGLang plugin dials SMG from inside it.
    assert cmd[1:3] == ["-m", "smg_grpc_servicer.sglang.headless"]
    assert "/models/qwen" in cmd and "--grpc-mode" not in cmd
    expected_port = serve._zmq_handshake_port(serve._zmq_ipc_url(w.port))
    assert cmd[cmd.index("--zmq-handshake-address") + 1] == f"tcp://127.0.0.1:{expected_port}"
    assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.5"


def test_sglang_zmq_extra_engine_args_go_through_the_launcher(serve):
    # Extras reach the launcher's validation (and its output) instead of
    # being appended behind its back: a DP launch the wire does not carry
    # is refused up front.
    w = _worker("sglang")
    w.extra_engine_args = ["--mem-fraction-static", "0.4"]
    cmd = w._build_cmd()
    assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.4"
    w.extra_engine_args = ["--dp-size", "2"]
    with pytest.raises(ValueError, match="dp-size"):
        w._build_cmd()


def test_tokenspeed_zmq_cmd_is_headless_with_derived_handshake_port(serve):
    w = _worker("tokenspeed")
    cmd = w._build_tokenspeed_zmq_cmd(
        "/models/qwen", 1, {"tokenspeed_args": ["--attention-backend", "fa3"]}
    )
    assert "serve" in cmd
    assert "--headless" in cmd
    assert "/models/qwen" in cmd
    expected_port = serve._zmq_handshake_port(serve._zmq_ipc_url(w.port))
    assert cmd[cmd.index("--data-parallel-rpc-port") + 1] == str(expected_port)
    assert cmd[cmd.index("--attention-backend") + 1] == "fa3"


def test_grpc_worker_still_uses_grpc_url():
    w = Worker(
        model_id=_VLLM_MODEL,
        engine="vllm",
        port=50111,
        gpu_ids=[0],
        mode=ConnectionMode.GRPC,
    )
    assert w.base_url == "grpc://127.0.0.1:50111"


def test_rust_servicer_impl_is_a_worker_env_flag(monkeypatch):
    """``E2E_VLLM_SERVICER_IMPL=rust`` keeps upstream's gRPC entrypoint as the
    worker command and selects the Rust path through the servicer package's
    flag in the worker environment, so every test case runs unchanged."""
    import infra.worker as worker_module
    from infra.constants import ENV_VLLM_SERVICER_IMPL

    hook_checks: list[bool] = []
    monkeypatch.setattr(
        worker_module, "_require_rust_servicer_hook", lambda: hook_checks.append(True)
    )

    w = Worker(
        model_id=_VLLM_MODEL,
        engine="vllm",
        port=50111,
        gpu_ids=[0],
        mode=ConnectionMode.GRPC,
    )
    spec = {"vllm_args": ["--max-model-len", "2048"]}

    monkeypatch.delenv(ENV_VLLM_SERVICER_IMPL, raising=False)
    monkeypatch.delenv("SMG_VLLM_SERVICER_IMPL", raising=False)
    assert "vllm.entrypoints.grpc_server" in w._build_vllm_grpc_cmd("/models/llama", 1, spec)
    assert "SMG_VLLM_SERVICER_IMPL" not in w._build_env()

    monkeypatch.setenv(ENV_VLLM_SERVICER_IMPL, "rust")
    cmd = w._build_vllm_grpc_cmd("/models/llama", 1, spec)
    assert "vllm.entrypoints.grpc_server" in cmd
    assert "--impl" not in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "2048"
    assert w._build_env()["SMG_VLLM_SERVICER_IMPL"] == "rust"
    # The Rust lane is only built once the installed vLLM is known to carry
    # the hook; without it the workers would silently run Python.
    assert hook_checks == [True]

    # The lane setting wins over a flag inherited from the operator's shell.
    monkeypatch.delenv(ENV_VLLM_SERVICER_IMPL, raising=False)
    monkeypatch.setenv("SMG_VLLM_SERVICER_IMPL", "rust")
    assert "SMG_VLLM_SERVICER_IMPL" not in w._build_env()

    monkeypatch.setenv(ENV_VLLM_SERVICER_IMPL, "go")
    with pytest.raises(ValueError, match=ENV_VLLM_SERVICER_IMPL):
        w._build_env()


def test_rust_lane_refuses_a_vllm_without_the_hook(monkeypatch):
    import infra.worker as worker_module
    from infra.constants import ENV_VLLM_SERVICER_IMPL

    def no_hook():
        raise RuntimeError("no hook")

    monkeypatch.setattr(worker_module, "_require_rust_servicer_hook", no_hook)
    monkeypatch.setenv(ENV_VLLM_SERVICER_IMPL, "rust")
    w = Worker(
        model_id=_VLLM_MODEL, engine="vllm", port=50111, gpu_ids=[0], mode=ConnectionMode.GRPC
    )
    with pytest.raises(RuntimeError, match="no hook"):
        w._build_env()


def test_vllm_workers_select_the_prebuilt_flashinfer_allreduce(monkeypatch):
    from infra.constants import ENV_VLLM_SERVICER_IMPL

    monkeypatch.delenv(ENV_VLLM_SERVICER_IMPL, raising=False)
    monkeypatch.delenv("SMG_VLLM_SERVICER_IMPL", raising=False)
    monkeypatch.delenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", raising=False)
    w = Worker(
        model_id=_VLLM_MODEL, engine="vllm", port=50111, gpu_ids=[0], mode=ConnectionMode.GRPC
    )
    # vLLM's auto backend picks the MNNVL kernel on Hopper, which FlashInfer's
    # per-architecture jit-cache wheel does not carry, and the pods cannot
    # JIT-compile it; the TRT-LLM kernel is prebuilt there.
    assert w._build_env()["VLLM_FLASHINFER_ALLREDUCE_BACKEND"] == "trtllm"

    # An operator's explicit choice wins over the harness default.
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl")
    assert w._build_env()["VLLM_FLASHINFER_ALLREDUCE_BACKEND"] == "mnnvl"

    # Other engines are not told anything about vLLM's all-reduce.
    monkeypatch.delenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", raising=False)
    s = Worker(
        model_id=_VLLM_MODEL, engine="sglang", port=50112, gpu_ids=[0], mode=ConnectionMode.GRPC
    )
    assert "VLLM_FLASHINFER_ALLREDUCE_BACKEND" not in s._build_env()
