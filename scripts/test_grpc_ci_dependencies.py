"""Exercise CI shells with fake package tools; no engine or network is used."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKE = r"""
import importlib.metadata
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with open(os.environ["TOOL_LOG"], "a") as log:
    log.write(json.dumps({"tool": Path(sys.argv[0]).name, "args": args}) + "\n")
if Path(sys.argv[0]).name == "pip" or args[:2] == ["-m", "pip"]:
    sys.exit(0)
if args[:2] == ["-m", "venv"]:
    path = Path(args[-1]) / "bin"
    path.mkdir(parents=True)
    (path / "activate").write_text("# fake activation\n")
elif args == ["-V"]:
    print("Python 3.12.0 (CPU control)")
elif args[0] == "-c" and args[1].lstrip().startswith("import cairo\n"):
    pass  # Cairo is already available in this companion-version shell fixture.
elif args[0] == "-c" and "importlib.metadata" in args[1]:
    def version(name):
        versions = json.loads(os.environ["IMAGE_VERSIONS"])
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]
    importlib.metadata.version = version
    exec(args[1])
elif args[0] == "-c" and "assert torch.version.hip" in args[1]:
    pass  # The real import guard stays in the shell; never load an engine here.
else:
    raise RuntimeError(args)
"""


@pytest.fixture
def tools(tmp_path):
    binary = tmp_path / "tools"
    binary.mkdir()
    for name in ["python3", "pip"]:
        path = binary / name
        path.write_text(f"#!{sys.executable}\n" + FAKE)
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "TOOL_LOG": str(tmp_path / "tools.jsonl"),
        "IMAGE_VERSIONS": json.dumps({"vllm": "0.31.0+rocm723", "grpcio-reflection": "1.78.0"}),
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "GITHUB_PATH": str(tmp_path / "github-path"),
    }
    env.pop("SMG_CI_GRPC_COMPANION_VERSION", None)
    Path(env["GITHUB_ENV"]).write_text("EXISTING=value\n")
    return tmp_path, env


def run_script(name, tools, extra_env=None):
    cwd, env = tools
    env = {**env, **(extra_env or {})}
    process = subprocess.run(
        ["bash", str(ROOT / "scripts" / name)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    calls = [json.loads(line) for line in Path(env["TOOL_LOG"]).read_text().splitlines()]
    return process, calls


@pytest.mark.parametrize(
    "engine,vendor", [("vllm", "nvidia"), ("sglang", "amd"), ("sglang", "nvidia")]
)
def test_other_jobs_keep_original_companions(tools, engine, vendor):
    process, calls = run_script(
        "ci_install_e2e_deps.sh", tools, {"E2E_ENGINE": engine, "E2E_GPU_VENDOR": vendor}
    )
    assert process.returncode == 0, process.stderr
    args = calls[-1]["args"]
    assert args[-2:] == ["grpcio-health-checking==1.81.*", "grpcio-reflection==1.81.*"]
    assert Path(tools[1]["GITHUB_ENV"]).read_text() == "EXISTING=value\n"


def test_amd_image_selection_reaches_later_job_step(tools):
    process, calls = run_script("ci_setup_vllm_rocm_image.sh", tools)
    assert process.returncode == 0, process.stderr
    assert any("vllm==0.31.0+rocm723" in call["args"] for call in calls)
    assert not any("--no-binary=pycairo" in call["args"] for call in calls)
    assert (
        Path(tools[1]["GITHUB_ENV"]).read_text()
        == "EXISTING=value\nSMG_CI_GRPC_COMPANION_VERSION=1.78.0\n"
    )
    value = Path(tools[1]["GITHUB_ENV"]).read_text().splitlines()[-1].split("=", 1)[1]
    process, calls = run_script(
        "ci_install_e2e_deps.sh", tools, {"SMG_CI_GRPC_COMPANION_VERSION": value}
    )
    assert process.returncode == 0, process.stderr
    assert calls[-1]["args"][-2:] == ["grpcio-health-checking==1.78.0", "grpcio-reflection==1.78.0"]


@pytest.mark.parametrize("missing", ["vllm", "grpcio-reflection"])
def test_missing_image_metadata_does_not_pollute_job(tools, missing):
    versions = json.loads(tools[1]["IMAGE_VERSIONS"])
    del versions[missing]
    process, calls = run_script(
        "ci_setup_vllm_rocm_image.sh", tools, {"IMAGE_VERSIONS": json.dumps(versions)}
    )
    assert process.returncode != 0 and "PackageNotFoundError" in process.stderr
    assert len(calls) == 1
    assert not (tools[0] / ".venv").exists()
    assert Path(tools[1]["GITHUB_ENV"]).read_text() == "EXISTING=value\n"
    assert not Path(tools[1]["GITHUB_PATH"]).exists()


def test_invalid_metadata_cannot_write_environment_lines(tools):
    process, calls = run_script(
        "ci_setup_vllm_rocm_image.sh",
        tools,
        {
            "IMAGE_VERSIONS": json.dumps(
                {"vllm": "0.31.0", "grpcio-reflection": "1.78.0\nINJECT=value"}
            )
        },
    )
    assert process.returncode != 0 and "InvalidVersion" in process.stderr
    assert len(calls) == 1
    assert Path(tools[1]["GITHUB_ENV"]).read_text() == "EXISTING=value\n"


def test_standalone_caller_can_explicitly_select_companions(tools):
    process, calls = run_script(
        "ci_install_e2e_deps.sh", tools, {"SMG_CI_GRPC_COMPANION_VERSION": "1.78.0"}
    )
    assert process.returncode == 0, process.stderr
    assert calls[-1]["args"][-2:] == ["grpcio-health-checking==1.78.0", "grpcio-reflection==1.78.0"]
