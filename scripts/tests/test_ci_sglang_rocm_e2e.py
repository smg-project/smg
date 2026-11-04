"""CPU provenance controls; no ROCm package or backend is imported."""

import importlib.util
import json
import os
import subprocess
import sys
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


@pytest.mark.parametrize(
    "changed", ["torch", "torchaudio", "torchvision", "requires", "constraint", "rpc"]
)
def test_persistent_setup_baseline_rejects_later_install_drift(
    metadata, rpc_sources, monkeypatch, changed
):
    versions, requirements, constraint = metadata
    root, _, _, modules = rpc_sources
    baseline = root / "baseline.json"
    monkeypatch.setattr(sys, "argv", ["verifier", "--snapshot", str(baseline)])
    VERIFIER.main()
    monkeypatch.setattr(
        sys, "argv", ["verifier", "--verify", str(baseline), "--capture-rpc", "--root", str(root)]
    )
    VERIFIER.main()
    frozen = baseline.read_bytes()
    if changed == "requires":
        requirements["torchvision"] = ["changed==2"]
    elif changed == "constraint":
        constraint.write_text("changed\n")
    elif changed == "rpc":
        modules["smg_grpc_servicer"].__file__ = "/wrong/origin.py"
    else:
        versions[changed] = "changed"
    monkeypatch.setattr(sys, "argv", ["verifier", "--verify", str(baseline), "--root", str(root)])
    with pytest.raises(RuntimeError, match="expected"):
        VERIFIER.main()
    assert baseline.read_bytes() == frozen, "a failed check must not reset the setup baseline"


def test_rpc_baseline_must_be_bound_once_and_cannot_refresh(metadata, rpc_sources, monkeypatch):
    root, _, _, _ = rpc_sources
    baseline = root / "baseline.json"
    monkeypatch.setattr(sys, "argv", ["verifier", "--snapshot", str(baseline)])
    VERIFIER.main()
    monkeypatch.setattr(sys, "argv", ["verifier", "--verify", str(baseline), "--root", str(root)])
    with pytest.raises(RuntimeError, match="RPC provenance differs"):
        VERIFIER.main()
    monkeypatch.setattr(
        sys, "argv", ["verifier", "--verify", str(baseline), "--capture-rpc", "--root", str(root)]
    )
    VERIFIER.main()
    frozen = baseline.read_bytes()
    with pytest.raises(RuntimeError, match="already captured"):
        VERIFIER.main()
    assert baseline.read_bytes() == frozen


@pytest.mark.parametrize("failed_stage", [None, "snapshot", "cohort", "verify", "pip"])
def test_real_setup_wrapper_preserves_order_failure_and_persistent_snapshot(tmp_path, failed_stage):
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
    raise SystemExit({"snapshot": 17, "cohort": 42, "verify": 47, "pip": 43}[stage])
if stage == "snapshot":
    pathlib.Path(args[-1]).write_text('{"torch": "fixture"}')
"""
    for name in ("python3", "bash"):
        stub = binary / name
        stub.write_text(dispatcher)
        stub.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    result = subprocess.run(
        ["/bin/bash", str(SOURCE.with_name("ci_setup_sglang_rocm_e2e.sh"))],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(binary) + ":/usr/bin:/bin",
            "WRAPPER_LOG": str(log),
            "FAILED_STAGE": failed_stage or "",
            "GITHUB_ENV": str(tmp_path / "github.env"),
            "SMG_SGLANG_ROCM_IMAGE": (
                "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
                "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
            ),
        },
        timeout=10,
    )
    assert (
        result.returncode
        == {None: 0, "snapshot": 17, "cohort": 42, "verify": 47, "pip": 43}[failed_stage]
    ), result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    order = ["snapshot", "cohort", "verify", "pip"]
    assert [command["stage"] for command in commands] == (
        order if failed_stage is None else order[: order.index(failed_stage) + 1]
    )
    snapshot = tmp_path / commands[0]["args"][-1]
    assert snapshot.exists() == (failed_stage != "snapshot")
    if failed_stage is None:
        assert "--capture-rpc" in commands[2]["args"]
        assert (tmp_path / "github.env").read_text() == (
            "SMG_CI_SGLANG_E2E_SNAPSHOT=e2e-logs/sglang-rocm-e2e-baseline.json\n"
        )


@pytest.mark.parametrize("requested", ["new-pin", "old-pin", "invalid-source"])
def test_wrapper_follows_authoritative_pin_before_any_python_or_setup(tmp_path, requested):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("ci_setup_sglang_rocm_e2e.sh", "ci_setup_sglang_rocm_image_pin.sh"):
        (scripts / name).write_bytes(SOURCE.with_name(name).read_bytes())
    old = "example/sglang:cohort@sha256:" + "a" * 64
    new = old[:-64] + "b" * 64
    declaration = f'IMAGE = "{new}"\n'
    if requested == "invalid-source":
        declaration += f'IMAGE = "{old}"\n'
    (scripts / "ci_verify_sglang_rocm_cohort.py").write_text(declaration)
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "mkdir").symlink_to("/usr/bin/mkdir")
    marker = tmp_path / "python-called"
    python = binary / "python3"
    python.write_text('#!/bin/bash\nprintf called > "$PROBE_MARKER"\nexit 73\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(scripts / "ci_setup_sglang_rocm_e2e.sh")],
        cwd=tmp_path,
        env={
            "PATH": str(binary),
            "PROBE_MARKER": str(marker),
            "SMG_SGLANG_ROCM_IMAGE": old if requested == "old-pin" else new,
        },
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == (73 if requested == "new-pin" else 1), result.stderr
    assert marker.exists() == (requested == "new-pin")
    if requested == "old-pin":
        assert f"observed {old}; expected {new}" in result.stderr


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
