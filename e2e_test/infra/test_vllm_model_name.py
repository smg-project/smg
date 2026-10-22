"""Keep the request model stable when vLLM resolves cached model weights."""

import pytest
from infra import worker as worker_module
from infra.constants import ConnectionMode
from infra.worker import Worker


@pytest.mark.parametrize("mode", [ConnectionMode.HTTP, ConnectionMode.GRPC])
@pytest.mark.parametrize("model", ["org/model", "/models/local/snapshot"])
def test_vllm_advertises_the_requested_model(monkeypatch, mode, model):
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"model": model})
    worker = Worker(model_id="unit-model", engine="vllm", port=50111, gpu_ids=[0], mode=mode)
    cmd = worker._build_cmd()
    assert cmd[cmd.index("--model") + 1] == model
    assert cmd[cmd.index("--served-model-name") + 1] == model
    assert cmd.count("--served-model-name") == 1


@pytest.mark.parametrize("location", ["spec", "worker"])
@pytest.mark.parametrize(
    "option", ["--served-model-name", "--served-model-name=alias", "--served_model_name"]
)
def test_explicit_vllm_model_names_remain_authoritative(monkeypatch, location, option):
    args = [option] if "=" in option else [option, "alias", "second-alias"]
    spec = {"model": "org/model", "vllm_args": args if location == "spec" else []}
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: spec)
    worker = Worker(
        model_id="unit-model",
        engine="vllm",
        port=50111,
        gpu_ids=[0],
        extra_engine_args=args if location == "worker" else None,
    )
    cmd = worker._build_cmd()
    assert cmd[-len(args) :] == args
    assert sum(arg.split("=", 1)[0].replace("_", "-") == "--served-model-name" for arg in cmd) == 1
    assert cmd[cmd.index("--model") + 1] == "org/model"
