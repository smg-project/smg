"""CPU provenance controls; no ROCm package or backend is imported."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SOURCE = Path(__file__).resolve().parents[1] / "ci_verify_sglang_rocm_e2e.py"
SPEC = importlib.util.spec_from_file_location("rocm_e2e_verifier", SOURCE)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    constraint = tmp_path / "torch-rocm.txt"
    constraint.write_text("torch==fixture\n")
    monkeypatch.setattr(VERIFIER, "CONSTRAINT", constraint)
    versions = {name: "fixture" for name in VERIFIER.TORCH_PACKAGES}
    requirements = {name: ["kept==1"] for name in VERIFIER.TORCH_PACKAGES}
    monkeypatch.setattr(VERIFIER.md, "version", versions.__getitem__)
    monkeypatch.setattr(VERIFIER.md, "requires", requirements.__getitem__)
    return versions, requirements, constraint


@pytest.mark.parametrize("changed", [None, "torch", "torchaudio", "torchvision", "constraint"])
def test_original_torch_metadata_and_constraint_are_preserved(metadata, changed):
    versions, requirements, constraint = metadata
    before = VERIFIER.torch_snapshot()
    if changed == "constraint":
        constraint.write_text("changed\n")
    elif changed == "torchaudio":
        requirements[changed] = ["changed==2"]
    elif changed:
        versions[changed] = "changed"
    if changed:
        with pytest.raises(RuntimeError, match="Torch packages or constraints changed"):
            VERIFIER.verify_torch_snapshot(before)
    else:
        VERIFIER.verify_torch_snapshot(before)


@pytest.fixture
def rpc_sources(tmp_path, monkeypatch):
    proto = tmp_path / "crates/grpc_client/proto"
    proto.mkdir(parents=True)
    (proto / "common.proto").touch()
    generated = tmp_path / "crates/grpc_client/python/smg_grpc_proto/generated"
    modules = {
        "smg_grpc_proto": SimpleNamespace(__file__=str(generated.parent / "__init__.py")),
        "smg_grpc_servicer": SimpleNamespace(
            __file__=str(tmp_path / "grpc_servicer/smg_grpc_servicer/__init__.py")
        ),
        "sglang": SimpleNamespace(__file__="/sgl-workspace/sglang/python/sglang/__init__.py"),
        "smg_grpc_proto.generated.common_pb2": SimpleNamespace(
            __file__=str(generated / "common_pb2.py")
        ),
        "smg_grpc_proto.generated.common_pb2_grpc": SimpleNamespace(
            __file__=str(generated / "common_pb2_grpc.py")
        ),
    }
    monkeypatch.setattr(VERIFIER.importlib, "import_module", modules.__getitem__)
    return tmp_path, proto, generated, modules


def test_generated_companions_follow_the_actual_proto_inventory(rpc_sources):
    root, proto, generated, modules = rpc_sources
    (proto / "new_engine.proto").touch()
    for suffix in ("_pb2", "_pb2_grpc"):
        modules["smg_grpc_proto.generated.new_engine" + suffix] = SimpleNamespace(
            __file__=str(generated / ("new_engine" + suffix + ".py"))
        )
    result = VERIFIER.verify_rpc_sources(root)
    assert len(result["generated_modules"]) == 4
    assert result["GPU_or_model_accepted"] is False


@pytest.mark.parametrize(
    "name",
    ["smg_grpc_proto", "smg_grpc_servicer", "sglang", "smg_grpc_proto.generated.common_pb2_grpc"],
)
def test_editable_and_generated_modules_cannot_resolve_elsewhere(rpc_sources, name):
    root, _, _, modules = rpc_sources
    modules[name].__file__ = "/wrong/source.py"
    with pytest.raises(RuntimeError, match="unexpected .* origin"):
        VERIFIER.verify_rpc_sources(root)


def test_a_new_proto_requires_both_generated_companions(rpc_sources):
    root, proto, _, _ = rpc_sources
    (proto / "unbuilt.proto").touch()
    with pytest.raises(KeyError, match="unbuilt_pb2"):
        VERIFIER.verify_rpc_sources(root)


def test_a_missing_proto_inventory_is_not_empty_success(rpc_sources):
    root, proto, _, _ = rpc_sources
    (proto / "common.proto").unlink()
    with pytest.raises(RuntimeError, match="No repository RPC proto"):
        VERIFIER.verify_rpc_sources(root)


@pytest.mark.parametrize("failed_stage", [None, "snapshot", "cohort", "verify"])
def test_real_setup_wrapper_preserves_order_failure_and_snapshot_cleanup(tmp_path, failed_stage):
    binary = tmp_path / "bin"
    binary.mkdir()
    dispatcher = """#!/usr/bin/python3
import json, os, pathlib, sys
args = sys.argv[1:]
if pathlib.Path(sys.argv[0]).name == "bash":
    assert pathlib.Path(args[0]).name == "ci_setup_sglang_rocm_image.sh"
    stage = "cohort"
elif "--snapshot" in args:
    stage = "snapshot"
elif "--verify" in args:
    stage = "verify"
else:
    assert args == ["-m", "pip", "check"]
    stage = "pip"
with open(os.environ["WRAPPER_LOG"], "a") as output:
    output.write(json.dumps({"stage": stage, "args": args}) + "\\n")
if stage == os.environ.get("FAILED_STAGE"):
    raise SystemExit({"snapshot": 17, "cohort": 42, "verify": 47}[stage])
"""
    for name in ("python3", "bash"):
        stub = binary / name
        stub.write_text(dispatcher)
        stub.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    result = subprocess.run(
        ["/bin/bash", str(SOURCE.with_name("ci_setup_sglang_rocm_e2e.sh"))],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(binary) + ":/usr/bin:/bin",
            "WRAPPER_LOG": str(log),
            "FAILED_STAGE": failed_stage or "",
            "SMG_SGLANG_ROCM_IMAGE": (
                "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
                "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
            ),
        },
        timeout=10,
    )
    assert (
        result.returncode == {None: 0, "snapshot": 17, "cohort": 42, "verify": 47}[failed_stage]
    ), result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    order = ["snapshot", "cohort", "verify", "pip"]
    assert [command["stage"] for command in commands] == (
        order if failed_stage is None else order[: order.index(failed_stage) + 1]
    )
    assert not Path(commands[0]["args"][-1]).exists()


@pytest.mark.parametrize(
    ("engine", "vendor", "image", "missing", "code"),
    [
        ("sglang", "amd", "pinned", False, 0),
        ("vllm", "amd", "pinned", False, 1),
        ("sglang", "nvidia", "pinned", False, 1),
        ("sglang", "amd", "wrong", False, 1),
        ("sglang", "amd", "pinned", True, 1),
    ],
)
def test_real_callee_scope_guard_requires_the_complete_fixed_image_route(
    tmp_path, engine, vendor, image, missing, code
):
    root = SOURCE.parents[1]
    jobs = yaml.safe_load((root / ".github/workflows/e2e-gpu-job.yml").read_text())["jobs"]
    assert len(jobs) == 1
    steps = next(iter(jobs.values()))["steps"]
    scope = next(
        step for step in steps if step["name"] == "Check final-environment verification scope"
    )
    setup = next(
        step for step in steps if step["name"] == "Setup AMD SGLang in its fixed ROCm image"
    )
    assert setup["run"] == "bash scripts/ci_setup_sglang_rocm_e2e.sh"
    assert setup["if"] == "inputs.final_environment_check"
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", scope["run"]],
        cwd=tmp_path if missing else root,
        env={
            **os.environ,
            "CHECK_ENGINE": engine,
            "CHECK_VENDOR": vendor,
            "CHECK_IMAGE": (
                "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
                "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
            )
            if image == "pinned"
            else image,
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == code, result.stderr
