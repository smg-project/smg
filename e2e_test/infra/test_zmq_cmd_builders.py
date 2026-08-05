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
