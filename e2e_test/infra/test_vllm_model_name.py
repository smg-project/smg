"""Match vLLM's advertised name to the model used by the test fixtures."""

import pytest
from infra import model_specs
from infra import worker as worker_module
from infra.constants import ConnectionMode
from infra.worker import Worker


@pytest.mark.parametrize("mode", [ConnectionMode.HTTP, ConnectionMode.GRPC])
@pytest.mark.parametrize("layout", ["hub_cache", "local_copy", "snapshot_symlink"])
def test_vllm_advertises_the_resolved_request_model(monkeypatch, tmp_path, mode, layout):
    model_id = "org/model"
    snapshot = tmp_path / "hub" / "models--org--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    local_root = tmp_path / "models"
    monkeypatch.setattr(model_specs, "ROUTER_LOCAL_MODEL_PATH", str(local_root))
    local_model = local_root / model_id
    if layout == "local_copy":
        local_model.mkdir(parents=True)
    elif layout == "snapshot_symlink":
        local_model.parent.mkdir(parents=True)
        local_model.symlink_to(snapshot, target_is_directory=True)
    resolved = model_specs._resolve_model_path(model_id)
    assert resolved == (model_id if layout == "hub_cache" else str(local_model))
    monkeypatch.setitem(model_specs.MODEL_SPECS, model_id, {"model": resolved})
    # setup_backend uses this same resolved spec value in requests.
    request_model = model_specs.get_model_spec(model_id)["model"]
    worker = Worker(model_id=model_id, engine="vllm", port=50111, gpu_ids=[0], mode=mode)
    cmd = worker._build_cmd()
    assert cmd[cmd.index("--model") + 1] == resolved
    assert cmd[cmd.index("--served-model-name") + 1] == request_model
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
