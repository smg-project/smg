"""Native encoder command construction; no engine import or subprocess startup."""

import pytest
from infra import worker as worker_module
from infra.constants import ConnectionMode, WorkerType
from infra.model_specs import get_model_spec
from infra.worker import Worker

MODEL = "Qwen/Qwen3.5-9B"


def encoder(**overrides):
    args = dict(
        model_id=MODEL,
        engine="sglang",
        port=50104,
        gpu_ids=[0],
        mode=ConnectionMode.GRPC,
        worker_type=WorkerType.ENCODE,
    )
    args.update(overrides)
    return Worker(**args)


def test_sglang_encoder_uses_native_rpc_and_scheduler_delivery():
    worker = encoder(extra_engine_args=["--context-length", "8192"])
    cmd = worker._build_cmd()
    assert cmd[:3] == ["python3", "-m", "sglang.launch_server"]
    assert cmd[cmd.index("--model-path") + 1] == get_model_spec(MODEL)["model"]
    assert cmd[cmd.index("--port") + 1] == "50104"
    assert cmd[cmd.index("--tp-size") + 1] == "1"
    assert cmd.count("--grpc-mode") == cmd.count("--encoder-only") == 1
    assert cmd[cmd.index("--encoder-transfer-backend") + 1] == "zmq_to_scheduler"
    assert "--disaggregation-mode" not in cmd
    assert "--language-only" not in cmd
    assert "fa3" not in cmd
    assert cmd[-2:] == ["--context-length", "8192"]
    assert worker.process is None


def test_text_only_encoder_rejected_before_environment_or_spawn(monkeypatch):
    worker = encoder(model_id="meta-llama/Llama-3.1-8B-Instruct")

    def unexpected(*args, **kwargs):
        pytest.fail("invalid encoder reached worker preparation")

    monkeypatch.setattr(worker, "_build_env", unexpected)
    monkeypatch.setattr(worker, "_spawn_process", unexpected)
    monkeypatch.setattr(worker_module, "gpu_memory_used_mib", unexpected)
    with pytest.raises(ValueError, match="require a multimodal model"):
        worker.start(wait_ready=False)
    assert worker.process is None


@pytest.mark.parametrize(
    "override",
    [
        ["--encoder-transfer-backend", "mooncake"],
        ["--encoder-transfer-backend=zmq_to_tokenizer"],
        ["--disaggregation-mode", "prefill"],
        ["--language-only"],
        ["--language-model-only"],
        ["--grpc-mode"],
    ],
)
def test_encoder_extras_cannot_replace_role_or_transfer(override):
    with pytest.raises(ValueError, match="flags are owned by Worker"):
        encoder(extra_engine_args=override)._build_cmd()


def test_model_options_cannot_override_native_encoder_transfer(monkeypatch):
    spec = {**get_model_spec(MODEL), "sglang_args": ["--encoder-transfer-backend=mooncake"]}
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: spec)
    with pytest.raises(ValueError, match="flags are owned by Worker"):
        encoder()._build_cmd()


def test_native_encoder_does_not_select_an_lm_servicer(monkeypatch):
    monkeypatch.setenv("SMG_SGLANG_SERVICER_IMPL", "rust")
    monkeypatch.setattr(worker_module, "get_sglang_servicer_impl", lambda: "rust")
    worker = encoder(extra_env={"SMG_SGLANG_SERVICER_IMPL": "rust"})
    assert "SMG_SGLANG_SERVICER_IMPL" not in worker._build_env()
    regular = encoder(worker_type=WorkerType.REGULAR)
    assert regular._build_env()["SMG_SGLANG_SERVICER_IMPL"] == "rust"
