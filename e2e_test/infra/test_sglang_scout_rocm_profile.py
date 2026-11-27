"""Real Worker command construction and vendor contracts; never start an engine."""

import copy

import pytest
from infra import constants
from infra.constants import ConnectionMode, WorkerType
from infra.model_specs import MODEL_SPECS
from infra.worker import Worker

SCOUT = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
GPTOSS120 = "openai/gpt-oss-120b"
PROFILE = "E2E_SGLANG_SCOUT_ROCM_ATTENTION_AUTO"


@pytest.fixture(autouse=True)
def _reference_lane(monkeypatch):
    monkeypatch.setenv("E2E_GPU_VENDOR", "nvidia")
    monkeypatch.delenv(PROFILE, raising=False)


def worker(model=SCOUT, **overrides):
    options = dict(
        model_id=model,
        engine="sglang",
        port=50131,
        gpu_ids=[0, 1, 2, 3],
        mode=ConnectionMode.GRPC,
        worker_type=WorkerType.REGULAR,
    )
    options.update(overrides)
    return Worker(**options)


def enable(monkeypatch, vendor="amd"):
    monkeypatch.setenv("E2E_GPU_VENDOR", vendor)
    monkeypatch.setenv(PROFILE, "1")


@pytest.mark.parametrize("vendor", ["amd", " AMD ", "AMD"])
def test_amd_scout_uses_sdk_attention_default(monkeypatch, vendor):
    actual = worker()
    mode_before, role_before = actual.mode, actual.worker_type
    before_specs = copy.deepcopy(MODEL_SPECS)
    reference = actual._build_cmd()
    assert "--attention-backend=fa3" in reference
    enable(monkeypatch, vendor)
    command = actual._build_cmd()
    assert command == [arg for arg in reference if arg != "--attention-backend=fa3"]
    assert MODEL_SPECS == before_specs
    assert actual.mode is mode_before
    assert actual.worker_type is role_before
    assert command[command.index("--tp-size") + 1] == "4"
    assert "--grpc-mode" in command


def test_profile_uses_existing_vendor_detection_contract(monkeypatch):
    actual = worker()
    monkeypatch.setenv(PROFILE, "1")
    monkeypatch.delenv("E2E_GPU_VENDOR")
    monkeypatch.setattr(
        constants.shutil, "which", lambda name: "/synthetic/amd-smi" if name == "amd-smi" else None
    )
    assert "--attention-backend=fa3" not in actual._build_cmd()


def test_active_profile_refuses_invalid_vendor(monkeypatch):
    enable(monkeypatch, "tpu")
    with pytest.raises(ValueError, match="E2E_GPU_VENDOR"):
        worker()._build_cmd()


def test_disabled_profile_does_not_change_vendor_validation(monkeypatch):
    actual = worker()
    reference = actual._build_cmd()
    monkeypatch.setenv("E2E_GPU_VENDOR", "tpu")
    assert actual._build_cmd() == reference


@pytest.mark.parametrize(
    "vendor,flag", [("nvidia", "1"), ("nvidia", ""), ("amd", ""), ("amd", "0")]
)
def test_reference_scout_keeps_fa3_without_both_amd_and_optin(monkeypatch, vendor, flag):
    actual = worker()
    reference = actual._build_cmd()
    monkeypatch.setenv("E2E_GPU_VENDOR", vendor)
    monkeypatch.setenv(PROFILE, flag)
    assert actual._build_cmd() == reference


def test_original_120b_arguments_are_unchanged(monkeypatch):
    actual = worker(GPTOSS120)
    before_specs = copy.deepcopy(MODEL_SPECS)
    reference = actual._build_cmd()
    enable(monkeypatch)
    assert actual._build_cmd() == reference
    assert MODEL_SPECS == before_specs


def test_other_engine_dispatch_is_unchanged(monkeypatch):
    actual = worker(engine="vllm")
    monkeypatch.setattr(actual, "_build_vllm_grpc_cmd", lambda *args: ["vllm-dispatch"])
    reference = actual._build_cmd()
    enable(monkeypatch)
    assert actual._build_cmd() == reference


@pytest.mark.parametrize("mode", [ConnectionMode.HTTP, ConnectionMode.ZMQ])
def test_profile_refuses_other_wire(monkeypatch, mode):
    enable(monkeypatch)
    with pytest.raises(ValueError, match="original TP4 gRPC"):
        worker(mode=mode)._build_cmd()


@pytest.mark.parametrize("role", [WorkerType.PREFILL, WorkerType.DECODE])
def test_profile_refuses_pd_roles(monkeypatch, role):
    enable(monkeypatch)
    with pytest.raises(ValueError, match="original TP4 gRPC"):
        worker(worker_type=role)._build_cmd()


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"tp": 2}, id="tp"),
        pytest.param({"extra_engine_args": ["--attention-backend", "fa3"]}, id="extra-arguments"),
    ],
)
def test_profile_refuses_other_tp_and_extra_arguments(monkeypatch, options):
    enable(monkeypatch)
    with pytest.raises(ValueError, match="original TP4 gRPC"):
        worker(**options)._build_cmd()


@pytest.mark.parametrize(
    "override",
    [
        [],
        ["--attention-backend=fa3", "--attention-backend=fa3"],
        ["--attention-backend=fa3", "--decode-attention-backend=triton"],
    ],
)
def test_profile_refuses_changed_attention_contract(monkeypatch, override):
    spec = copy.deepcopy(MODEL_SPECS[SCOUT])
    spec["sglang_args"] = [
        arg for arg in spec["sglang_args"] if arg != "--attention-backend=fa3"
    ] + override
    monkeypatch.setitem(MODEL_SPECS, SCOUT, spec)
    enable(monkeypatch)
    with pytest.raises(ValueError, match="reference FA3 override"):
        worker()._build_cmd()
