"""Run the ROCm setup's real import preflight on controlled Python packages.

These fixtures prove API/provenance rejection and cross-step environment
handoff. They do not qualify the image, native libraries, UCX or GPUs.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci_setup_sglang_rocm_image.sh"
API = "nixl_agent nixl_agent_config nixl_thread_sync_t".split()
BINDINGS = "nixlBackendError nixlCancelledError nixlRemoteDisconnectError".split()


def run_setup(tmp_path, *, backend="nixl", legacy="", hip="fixture", alias=True, missing=""):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "torch.py").write_text(
        "from types import SimpleNamespace\nversion = SimpleNamespace(hip=" + repr(hip) + ")\n"
    )
    for name in ("sglang", "mooncake", "smg_grpc_servicer"):
        (modules / f"{name}.py").write_text('__version__ = "fixture"\n')
    reflection = modules / "grpcio_reflection-1.84.0.dist-info"
    reflection.mkdir()
    (reflection / "METADATA").write_text("Name: grpcio-reflection\nVersion: 1.84.0\n")
    rocm = modules / "nixl_rocm"
    rocm.mkdir()
    (rocm / "__init__.py").write_text("")
    for file, names in (("_api.py", API), ("_bindings.py", BINDINGS)):
        (rocm / file).write_text(
            "".join(f"{name} = object()\n" for name in names if name != missing)
        )
    if alias:
        (modules / "nixl").symlink_to(rocm, target_is_directory=True)
    else:
        unrelated = modules / "nixl"
        unrelated.mkdir()
        (unrelated / "__init__.py").write_text("")
        # Keep every API otherwise valid: only the provenance guard rejects
        # this package, not a later missing-symbol import.
        for file in ("_api.py", "_bindings.py"):
            (unrelated / file).write_bytes((rocm / file).read_bytes())
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['-m', 'pip']:\n"
        "    with open(os.environ['PIP_LOG'], 'a') as out:\n"
        "        out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "else:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(modules),
        "E2E_KV_BACKEND": backend,
        "E2E_SGLANG_TRANSFER_BACKEND": legacy,
        "LD_LIBRARY_PATH": "/image/rocm/lib",
        "GITHUB_ENV": str(tmp_path / "github.env"),
        "PIP_LOG": str(tmp_path / "pip.log"),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
    )
    return result, env


@pytest.mark.parametrize("backend,legacy", [("nixl", ""), ("", "nixl")])
def test_rocm_nixl_preflight_and_next_step_loader_path(tmp_path, backend, legacy):
    result, env = run_setup(tmp_path, backend=backend, legacy=legacy)
    assert result.returncode == 0, result.stderr
    assert "SGLang ROCm NIXL import canary OK" in result.stdout
    exports = dict(line.split("=", 1) for line in Path(env["GITHUB_ENV"]).read_text().splitlines())
    assert exports["LD_LIBRARY_PATH"] == "/opt/ucx/lib:/image/rocm/lib"
    # A separate process consumes the producer's actual environment file.
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ['LD_LIBRARY_PATH'])"],
        env={**env, **exports},
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip() == "/opt/ucx/lib:/image/rocm/lib"
    assert "nixl" not in Path(env["PIP_LOG"]).read_text()


@pytest.mark.parametrize("backend,legacy", [("", ""), ("mooncake", "nixl")])
def test_default_and_explicit_mooncake_do_not_probe_nixl(tmp_path, backend, legacy):
    result, env = run_setup(tmp_path, backend=backend, legacy=legacy, alias=False, missing=API[0])
    assert result.returncode == 0, result.stderr
    assert "NIXL import canary" not in result.stdout
    assert "LD_LIBRARY_PATH=" not in Path(env["GITHUB_ENV"]).read_text()


@pytest.mark.parametrize("missing", API + BINDINGS)
def test_incompatible_sglang_nixl_api_stops_before_pip(tmp_path, missing):
    result, env = run_setup(tmp_path, missing=missing)
    assert result.returncode != 0
    assert missing in result.stderr
    assert not Path(env["PIP_LOG"]).exists()
    assert not Path(env["GITHUB_ENV"]).exists()


def test_plain_nixl_package_cannot_masquerade_as_the_rocm_backend(tmp_path):
    result, env = run_setup(tmp_path, alias=False)
    assert result.returncode != 0
    assert "does not resolve to the image's nixl_rocm backend" in result.stderr
    assert not Path(env["PIP_LOG"]).exists()


def test_cuda_torch_stops_before_nixl_or_pip(tmp_path):
    result, env = run_setup(tmp_path, hip=None)
    assert result.returncode != 0
    assert "requires the ROCm image's HIP torch" in result.stderr
    assert not Path(env["PIP_LOG"]).exists()
