"""Verify installer HTTP requirements and ROCm image companion handoff."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
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
        value = os.environ.get("IMAGE_REFLECTION_VERSION")
        if name != "grpcio-reflection" or value is None:
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
    if "import torch, sglang, mooncake, smg_grpc_servicer" in args[1]:
        for name in ("torch", "sglang", "mooncake", "smg_grpc_servicer"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["torch"].version = types.SimpleNamespace(hip="fixture")
        sys.modules["sglang"].__version__ = "fixture"
    exec(args[1])
elif args[:2] == ["-m", "pip"]:
    with open(os.environ["TOOL_LOG"], "a") as log:
        log.write(json.dumps(args) + "\n")
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
    log = tmp_path / "pip.jsonl"
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "ENGINE_REQUIREMENTS": json.dumps(packages),
        "TOOL_LOG": str(log),
        "BROKEN_METADATA": "1" if broken else "",
        "GITHUB_ENV": str(tmp_path / "github.env"),
    }
    for name in ("SMG_CI_GRPC_COMPANION_VERSION", "IMAGE_REFLECTION_VERSION"):
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
            "-e",
            "crates/grpc_client/python/",
            "-e",
            "grpc_servicer/",
        ]
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
