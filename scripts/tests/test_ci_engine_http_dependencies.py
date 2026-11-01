"""Verify installer HTTP requirements and ROCm image companion handoff."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_SOURCE = (
    "torch-c-dlpack-ext @ https://files.pythonhosted.org/packages/37/de/"
    "921b6491efce5c389a5ef9bbed3d2d6660005840dae488124173180859ab/"
    "torch_c_dlpack_ext-0.1.5.tar.gz#sha256="
    "d06f0357d575d22a168cc77acb9020fc4bae30968ceb6718a055dcbe92bacabe"
)
FAKE = r"""
import importlib.metadata
import json
import os
import sys
import types

args = sys.argv[1:]
if args == ["-V"]:
    print("Python (test fixture)")
elif args[0] == "-c":
    def version(name):
        value = {
            "grpcio-reflection": os.environ.get("IMAGE_REFLECTION_VERSION"),
            "sglang": os.environ.get("IMAGE_SGLANG_VERSION", "0.5.21"),
            "torch": os.environ.get("IMAGE_TORCH_VERSION", "2.11.0+rocm10.0.0"),
            "apache-tvm-ffi": "0.1.12",
            "tilelang": "0.1.15",
            "torch-c-dlpack-ext": "0.1.5",
        }.get(name)
        if value is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return value

    def requires(name):
        if os.environ.get("BROKEN_METADATA"):
            raise ValueError("unreadable engine metadata")
        packages = json.loads(os.environ["ENGINE_REQUIREMENTS"])
        if name not in packages:
            raise importlib.metadata.PackageNotFoundError(name)
        return packages[name]
    importlib.metadata.requires = requires
    importlib.metadata.version = version
    if "import torch" in args[1]:
        torch = types.ModuleType("torch")
        torch.version = types.SimpleNamespace(hip=os.environ.get("IMAGE_TORCH_HIP", "fixture"))
        torch.Tensor = type("Tensor", (), {})
        mode = os.environ.get("IMAGE_TORCH_C_API", "present")
        if mode != "absent":
            # Software fixture only: this is not a usable Torch C API capsule.
            torch.Tensor.__dlpack_c_exchange_api__ = (
                type("PyCapsule", (), {})() if mode == "present" else object()
            )
        sys.modules["torch"] = torch
    if "import torch, sglang, mooncake, smg_grpc_servicer" in args[1]:
        if os.environ.get("FAIL_NATIVE_CHECK"):
            raise SystemExit(44)
        for name in ("sglang", "mooncake", "smg_grpc_servicer", "tvm_ffi", "tilelang"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["sglang"].__version__ = "fixture"
        extension = types.ModuleType("torch_c_dlpack_ext")
        extension.core = types.SimpleNamespace(_lib=None)
        sys.modules["torch_c_dlpack_ext"] = extension
    exec(args[1])
elif args[:2] == ["-m", "pip"]:
    with open(os.environ["TOOL_LOG"], "a") as log:
        log.write(json.dumps(args) + "\n")
    with open(os.environ["TOOL_ENV_LOG"], "a") as log:
        log.write(json.dumps({key: os.environ.get(key) for key in (
            "PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT", "UV_CONSTRAINT"
        )}) + "\n")
    stage = os.environ.get("FAIL_PIP_STAGE")
    if stage == "cohort" and "apache-tvm-ffi==0.1.12" in args:
        raise SystemExit(42)
    if stage == "check" and args == ["-m", "pip", "check"]:
        raise SystemExit(43)
else:
    raise RuntimeError(args)
"""


def run_installer(
    tmp_path, packages, *, broken=False, script="ci_install_e2e_deps.sh", extra_env=None
):
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    python = binary / "python3"
    python.write_text(f"#!{sys.executable}\n" + FAKE)
    python.chmod(0o755)
    # Run the real retry wrapper without spending its backoff delays in controls.
    sleep = binary / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    log = tmp_path / "pip.jsonl"
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "ENGINE_REQUIREMENTS": json.dumps(packages),
        "TOOL_LOG": str(log),
        "TOOL_ENV_LOG": str(tmp_path / "pip-env.jsonl"),
        "BROKEN_METADATA": "1" if broken else "",
        "GITHUB_ENV": str(tmp_path / "github.env"),
    }
    for name in (
        "SMG_CI_GRPC_COMPANION_VERSION",
        "IMAGE_REFLECTION_VERSION",
        "IMAGE_SGLANG_VERSION",
        "IMAGE_TORCH_VERSION",
        "IMAGE_TORCH_HIP",
        "IMAGE_TORCH_C_API",
        "FAIL_PIP_STAGE",
        "FAIL_NATIVE_CHECK",
    ):
        env.pop(name, None)
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, [
        json.loads(line) for line in log.read_text().splitlines()
    ] if log.exists() else []


@pytest.mark.parametrize(
    "engine,requirements",
    [
        ("tensorrt-llm", ["mcp<2", "fastapi<=0.121.3,>=0.120.1", "starlette>=0.49.1"]),
        ("vllm", ["mcp>=2,<3", "fastapi[standard]>=0.133,<0.137", "starlette>=1.0.1"]),
    ],
)
def test_engine_requirements_join_the_e2e_transaction(tmp_path, engine, requirements):
    result, calls = run_installer(tmp_path, {engine: [*requirements, "fastapi-cli", "torch"]})
    assert result.returncode == 0, result.stderr
    assert calls[0] == ["-m", "pip", "install", "e2e_test/", *requirements]


def test_without_an_installed_engine_the_existing_install_is_preserved(tmp_path):
    result, calls = run_installer(tmp_path, {})
    assert result.returncode == 0, result.stderr
    assert calls[0] == ["-m", "pip", "install", "e2e_test/"]
    assert calls[-1] == [
        "-m",
        "pip",
        "install",
        "grpcio-health-checking==1.81.*",
        "grpcio-reflection==1.81.*",
    ]


def test_unreadable_engine_metadata_stops_before_pip(tmp_path):
    result, calls = run_installer(tmp_path, {}, broken=True)
    assert result.returncode != 0 and "unreadable engine metadata" in result.stderr
    assert calls == []


@pytest.mark.parametrize("image_version", ["1.84.0", "1.90.0"])
def test_sglang_image_companion_survives_separate_actions_steps(tmp_path, image_version):
    # Metadata/native modules/pip are fixtures here; this proves only shell wiring.
    github_env = tmp_path / "github.env"
    github_env.write_text("EXISTING=value\n")
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={"IMAGE_REFLECTION_VERSION": image_version},
    )
    assert result.returncode == 0, result.stderr
    assert calls == [
        [
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-build-isolation",
            "--no-binary=torch-c-dlpack-ext",
            "apache-tvm-ffi==0.1.12",
            "tilelang==0.1.15",
            EXTENSION_SOURCE,
        ],
        ["-m", "pip", "install", "-e", "crates/grpc_client/python/", "-e", "grpc_servicer/"],
        ["-m", "pip", "check"],
    ]
    assert github_env.read_text() == (
        "EXISTING=value\nSMG_CI_GRPC_COMPANION_VERSION=" + image_version + "\n"
    )
    name, value = github_env.read_text().splitlines()[-1].split("=", 1)
    (tmp_path / "pip.jsonl").unlink()
    # A new bash process consumes the real producer output, not the setup shell.
    result, calls = run_installer(tmp_path, {}, extra_env={name: value})
    assert result.returncode == 0, result.stderr
    assert calls[-1] == [
        "-m",
        "pip",
        "install",
        "grpcio-health-checking==" + image_version,
        "grpcio-reflection==" + image_version,
    ]


def test_missing_sglang_image_companion_stops_before_install(tmp_path):
    result, calls = run_installer(tmp_path, {}, script="ci_setup_sglang_rocm_image.sh")
    assert result.returncode != 0
    assert "SGLang ROCm image lacks grpcio-reflection metadata" in result.stderr
    assert calls == []
    assert not (tmp_path / "github.env").exists()


@pytest.mark.parametrize(
    "bad_image",
    [
        {"IMAGE_SGLANG_VERSION": "0.5.22"},
        {"IMAGE_TORCH_VERSION": "2.13.0"},
        {"IMAGE_TORCH_HIP": ""},
        {"IMAGE_TORCH_C_API": "absent"},
        {"IMAGE_TORCH_C_API": "invalid"},
    ],
)
def test_unsupported_rocm_cohort_stops_before_pip(tmp_path, bad_image):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={"IMAGE_REFLECTION_VERSION": "1.84.0", **bad_image},
    )
    assert result.returncode != 0
    assert "ERROR:" in result.stderr
    assert calls == [] and not (tmp_path / "github.env").exists()


@pytest.mark.parametrize("stage,exitcode", [("cohort", 42), ("native", 44), ("check", 43)])
def test_rocm_repair_failure_does_not_publish_handoff(tmp_path, stage, exitcode):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "FAIL_PIP_STAGE": stage,
            "FAIL_NATIVE_CHECK": "1" if stage == "native" else "",
        },
    )
    assert result.returncode == exitcode
    assert not (tmp_path / "github.env").exists()
    if stage == "cohort":
        assert len(calls) == 3
        assert all("apache-tvm-ffi==0.1.12" in call for call in calls)
    else:
        assert len(calls) == (2 if stage == "native" else 3)
        assert calls[1] == [
            "-m",
            "pip",
            "install",
            "-e",
            "crates/grpc_client/python/",
            "-e",
            "grpc_servicer/",
        ]


def test_rocm_repair_preserves_all_constraint_inputs(tmp_path):
    contents = "torch==2.11.0+rocm10.0.0\n"
    paths = {
        name: tmp_path / name
        for name in ("PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT", "UV_CONSTRAINT")
    }
    for path in paths.values():
        path.write_text(contents)
    expected = {name: str(path) for name, path in paths.items()}
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={"IMAGE_REFLECTION_VERSION": "1.84.0", **expected},
    )
    assert result.returncode == 0, result.stderr
    observed = [json.loads(line) for line in (tmp_path / "pip-env.jsonl").read_text().splitlines()]
    assert observed == [expected] * len(calls)
    assert all(path.read_text() == contents for path in paths.values())


def test_generic_cuda_install_does_not_repair_the_rocm_cohort(tmp_path):
    result, calls = run_installer(tmp_path, {"sglang": ["fastapi>=0.120.1"]})
    assert result.returncode == 0, result.stderr
    assert calls[0] == ["-m", "pip", "install", "e2e_test/", "fastapi>=0.120.1"]
    assert all(
        "apache-tvm-ffi==0.1.12" not in call
        and "tilelang==0.1.15" not in call
        and EXTENSION_SOURCE not in call
        for call in calls
    )
    assert not (tmp_path / "github.env").exists()
