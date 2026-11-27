"""CPU command tests: use the real builder AST without importing worker/SDK code."""

import ast
import copy
import os
from enum import Enum
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCOUT = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
GPTOSS120 = "openai/gpt-oss-120b"
PROFILE = "E2E_SGLANG_SCOUT_ROCM_ATTENTION_AUTO"


class Mode(Enum):
    HTTP = "http"
    GRPC = "grpc"
    ZMQ = "zmq"


class Role(Enum):
    REGULAR = "regular"
    PREFILL = "prefill"
    DECODE = "decode"


def model_specs():
    tree = ast.parse((ROOT / "e2e_test/infra/model_specs.py").read_text())
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign):
            target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "MODEL_SPECS":
            continue
        specs = {}
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            model = ast.literal_eval(key)
            if model not in {SCOUT, GPTOSS120}:
                continue
            spec = {"model": model}
            for field, data in zip(value.keys, value.values, strict=True):
                name = ast.literal_eval(field)
                if name in {"tp", "features", "sglang_args"}:
                    spec[name] = ast.literal_eval(data)
            specs[model] = spec
        assert set(specs) == {SCOUT, GPTOSS120}
        return specs
    raise AssertionError("actual MODEL_SPECS definition missing")


@pytest.fixture
def commands(monkeypatch):
    monkeypatch.delenv("E2E_GPU_VENDOR", raising=False)
    monkeypatch.delenv(PROFILE, raising=False)
    path = ROOT / "e2e_test/infra/worker.py"
    tree = ast.parse(path.read_text())
    source = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Worker"
    )
    cls = ast.parse("class WorkerCommands: pass").body[0]
    cls.body = [
        copy.deepcopy(node)
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_build_cmd", "_build_sglang_cmd"}
    ]
    assert len(cls.body) == 2
    specs = model_specs()
    namespace = {
        "os": os,
        "ConnectionMode": Mode,
        "WorkerType": Role,
        "DEFAULT_HOST": "127.0.0.1",
        "get_model_spec": specs.__getitem__,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"
        ),
        namespace,
    )

    def make(model=SCOUT):
        worker = namespace["WorkerCommands"]()
        worker.model_id = model
        worker.engine = "sglang"
        worker.mode = Mode.GRPC
        worker.worker_type = Role.REGULAR
        worker.port = 31000
        worker.tp = None
        worker.extra_engine_args = None
        worker.bootstrap_port = None
        worker.ib_device = None
        worker.effective_kv_backend = lambda: "mooncake"
        return worker

    return make, specs


def enable(monkeypatch):
    monkeypatch.setenv("E2E_GPU_VENDOR", "amd")
    monkeypatch.setenv(PROFILE, "1")


def test_amd_scout_uses_sdk_attention_default(commands, monkeypatch):
    make, specs = commands
    worker = make()
    before_specs = copy.deepcopy(specs)
    reference = worker._build_cmd()
    assert "--attention-backend=fa3" in reference
    enable(monkeypatch)
    assert worker._build_cmd() == [arg for arg in reference if arg != "--attention-backend=fa3"]
    assert specs == before_specs
    assert worker._build_cmd()[worker._build_cmd().index("--tp-size") + 1] == "4"
    assert "--grpc-mode" in worker._build_cmd()


@pytest.mark.parametrize(
    "vendor,flag", [("nvidia", "1"), ("nvidia", ""), ("amd", ""), ("amd", "0")]
)
def test_reference_scout_keeps_fa3_without_both_amd_and_optin(commands, monkeypatch, vendor, flag):
    make, _ = commands
    worker = make()
    reference = worker._build_cmd()
    monkeypatch.setenv("E2E_GPU_VENDOR", vendor)
    monkeypatch.setenv(PROFILE, flag)
    assert worker._build_cmd() == reference


def test_original_120b_arguments_are_unchanged(commands, monkeypatch):
    make, specs = commands
    worker = make(GPTOSS120)
    before_specs = copy.deepcopy(specs)
    reference = worker._build_cmd()
    enable(monkeypatch)
    assert worker._build_cmd() == reference
    assert specs == before_specs


def test_other_engine_is_unchanged(commands, monkeypatch):
    make, _ = commands
    worker = make()
    worker.engine = "vllm"
    worker._build_vllm_grpc_cmd = lambda *args: ["actual-vllm-dispatch-sentinel"]
    reference = worker._build_cmd()
    enable(monkeypatch)
    assert worker._build_cmd() == reference


@pytest.mark.parametrize("mode", [Mode.HTTP, Mode.ZMQ])
def test_profile_refuses_other_wire(commands, monkeypatch, mode):
    make, _ = commands
    worker = make()
    worker.mode = mode
    enable(monkeypatch)
    with pytest.raises(ValueError, match="original TP4 gRPC"):
        worker._build_cmd()


@pytest.mark.parametrize("role", [Role.PREFILL, Role.DECODE])
def test_profile_refuses_pd_roles(commands, monkeypatch, role):
    make, _ = commands
    worker = make()
    worker.worker_type = role
    enable(monkeypatch)
    with pytest.raises(ValueError, match="original TP4 gRPC"):
        worker._build_cmd()


def test_profile_refuses_other_tp(commands, monkeypatch):
    make, _ = commands
    worker = make()
    worker.tp = 2
    enable(monkeypatch)
    with pytest.raises(ValueError, match="original TP4 gRPC"):
        worker._build_cmd()


def test_profile_refuses_extra_engine_arguments(commands, monkeypatch):
    make, _ = commands
    worker = make()
    worker.extra_engine_args = ["--attention-backend", "fa3"]
    enable(monkeypatch)
    with pytest.raises(ValueError, match="original TP4 gRPC"):
        worker._build_cmd()


@pytest.mark.parametrize(
    "override",
    [
        [],
        ["--attention-backend=fa3", "--attention-backend=fa3"],
        ["--attention-backend=fa3", "--decode-attention-backend=triton"],
    ],
)
def test_profile_refuses_changed_attention_contract(commands, monkeypatch, override):
    make, specs = commands
    args = specs[SCOUT]["sglang_args"]
    specs[SCOUT]["sglang_args"] = [
        arg for arg in args if arg != "--attention-backend=fa3"
    ] + override
    enable(monkeypatch)
    with pytest.raises(ValueError, match="reference FA3 override"):
        make()._build_cmd()
