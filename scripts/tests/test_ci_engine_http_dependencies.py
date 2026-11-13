"""Verify installer HTTP requirements and ROCm image companion handoff."""

import base64
import hashlib
import importlib.util
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
import importlib.util
import json
import os
import sys
import types

args = sys.argv[1:]
if args[0].endswith("ci_verify_sglang_rocm_cohort.py"):
    with open(os.environ["TOOL_NATIVE_LOG"], "a") as log:
        log.write(json.dumps(args) + "\n")
    if os.environ.get("FAIL_NATIVE_CHECK"):
        raise SystemExit(44)
    # Run the real builtin rejection logic with native objects replaced by
    # software fixtures. The remaining full-cohort checks are native acceptance,
    # outside this shell-wiring test; RECORD is exercised separately below.
    spec = importlib.util.spec_from_file_location("cohort_verifier", args[0])
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    api = object()
    torch = types.SimpleNamespace(Tensor=types.SimpleNamespace(__dlpack_c_exchange_api__=api))
    core = types.SimpleNamespace(_lib=os.environ.get("IMAGE_DLPACK_LIB"),
                                 load_torch_c_dlpack_extension=lambda: None)
    verifier.verify_builtin_dlpack(core, torch, api)
    if len(args) == 2 and os.environ.get("FAIL_PIP_STAGE") != "check":
        with open(args[1], "w") as report:
            json.dump({"verified": True, "fixture": True}, report)
    args = ["-m", "pip", "check"]
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
    exec(args[1])
elif args[:2] == ["-m", "pip"]:
    known = (
        args == ["-m", "pip", "check"]
        or args == ["-m", "pip", "install", "clients/python/"]
        or (args[:4] == ["-m", "pip", "install", "e2e_test/"]
            and all(arg.startswith(("mcp", "fastapi", "starlette")) for arg in args[4:]))
        or (args[:3] == ["-m", "pip", "install"] and len(args) == 5
            and args[3].startswith("grpcio-health-checking==")
            and args[4].startswith("grpcio-reflection=="))
        or args == ["-m", "pip", "install", "-e", "crates/grpc_client/python/", "-e", "grpc_servicer/"]
        or (args[:8] == ["-m", "pip", "install", "--no-cache-dir", "--no-build-isolation",
                        "--no-binary=torch-c-dlpack-ext", "apache-tvm-ffi==0.1.12", "tilelang==0.1.15"]
            and len(args) == 9 and args[8].startswith("torch-c-dlpack-ext @ https://files.pythonhosted.org/"))
    )
    if not known:
        raise RuntimeError("unknown fixture pip command: " + repr(args))
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
    tmp_path,
    packages,
    *,
    broken=False,
    script="ci_install_e2e_deps.sh",
    extra_env=None,
    arguments=(),
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
        "TOOL_NATIVE_LOG": str(tmp_path / "native.jsonl"),
        "BROKEN_METADATA": "1" if broken else "",
        "GITHUB_ENV": str(tmp_path / "github.env"),
    }
    for name in (
        "SMG_CI_GRPC_COMPANION_VERSION",
        "SMG_SGLANG_ROCM_IMAGE",
        "IMAGE_REFLECTION_VERSION",
        "IMAGE_SGLANG_VERSION",
        "IMAGE_TORCH_VERSION",
        "IMAGE_TORCH_HIP",
        "IMAGE_TORCH_C_API",
        "IMAGE_DLPACK_LIB",
        "FAIL_PIP_STAGE",
        "FAIL_NATIVE_CHECK",
    ):
        env.pop(name, None)
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *arguments],
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
    native = [json.loads(line) for line in (tmp_path / "native.jsonl").read_text().splitlines()]
    assert native == [[str(ROOT / "scripts/ci_verify_sglang_rocm_cohort.py")]]
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
    "bad_image,diagnostic",
    [
        ({"IMAGE_SGLANG_VERSION": "0.5.22"}, "observed sglang=0.5.22"),
        ({"IMAGE_TORCH_VERSION": "2.13.0"}, "torch=2.13.0"),
        ({"IMAGE_TORCH_HIP": ""}, "hip=''"),
        ({"IMAGE_TORCH_C_API": "absent"}, "expected PyCapsule, observed NoneType"),
        ({"IMAGE_TORCH_C_API": "invalid"}, "expected PyCapsule, observed object"),
    ],
)
def test_unsupported_rocm_cohort_stops_before_pip(tmp_path, bad_image, diagnostic):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={"IMAGE_REFLECTION_VERSION": "1.84.0", **bad_image},
    )
    assert result.returncode != 0
    assert "ERROR:" in result.stderr
    assert diagnostic in result.stderr
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


FIXED_IMAGE = (
    "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
    "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
)


def test_verify_only_does_not_install_or_publish_handoff(tmp_path):
    handoff = tmp_path / "github.env"
    handoff.write_text("EXISTING=value\n")
    report = tmp_path / "native.json"
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=("--verify-only", str(report)),
        extra_env={"IMAGE_REFLECTION_VERSION": "1.84.0", "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE},
    )
    assert result.returncode == 0, result.stderr
    assert calls == [["-m", "pip", "check"]]
    assert handoff.read_text() == "EXISTING=value\n"
    assert json.loads(report.read_text())["verified"] is True
    native = [json.loads(line) for line in (tmp_path / "native.jsonl").read_text().splitlines()]
    assert native == [[str(ROOT / "scripts/ci_verify_sglang_rocm_cohort.py"), str(report)]]


def test_verify_only_keeps_a_failed_check_nonzero(tmp_path):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=("--verify-only", str(tmp_path / "native.json")),
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
            "FAIL_PIP_STAGE": "check",
        },
    )
    assert result.returncode == 43
    assert calls == [["-m", "pip", "check"]]
    assert not (tmp_path / "github.env").exists()


@pytest.mark.parametrize("state", ["valid", "modified", "escape"])
def test_cohort_record_requires_matching_bodies_and_contained_paths(tmp_path, state):
    spec = importlib.util.spec_from_file_location(
        "cohort_verifier", ROOT / "scripts/ci_verify_sglang_rocm_cohort.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    root = tmp_path / "site-packages"
    folder = root / "fixture.dist-info"
    folder.mkdir(parents=True)
    body = b"original payload\n"
    (root / "payload.py").write_bytes(body)
    encoded = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
    name = "payload.py"
    if state == "modified":
        (root / name).write_bytes(b"modified payload\n")
    elif state == "escape":
        name = "../outside.py"
        (tmp_path / "outside.py").write_bytes(body)
    (folder / "RECORD").write_text(
        f"{name},sha256={encoded},{len(body)}\nfixture.dist-info/RECORD,,\n"
    )
    if state == "valid":
        assert verifier.verify_record(root, folder) == {"payload.py", "fixture.dist-info/RECORD"}
    else:
        with pytest.raises(RuntimeError):
            verifier.verify_record(root, folder)


def test_dlpack_own_library_is_rejected_before_check_and_handoff(tmp_path):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "IMAGE_DLPACK_LIB": "fixture-dlpack-library",
        },
    )
    assert result.returncode != 0
    assert "DLPack extension loaded its own library" in result.stderr
    assert "fixture-dlpack-library" in result.stderr
    assert len(calls) == 2
    assert not (tmp_path / "github.env").exists()


def load_cohort_verifier():
    spec = importlib.util.spec_from_file_location(
        "cohort_verifier", ROOT / "scripts/ci_verify_sglang_rocm_cohort.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier


@pytest.mark.parametrize("image", [None, "different-image"])
def test_verify_only_rejects_missing_or_different_image_before_tools(tmp_path, image):
    extra = {"IMAGE_REFLECTION_VERSION": "1.84.0"}
    if image is not None:
        extra["SMG_SGLANG_ROCM_IMAGE"] = image
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=("--verify-only",),
        extra_env=extra,
    )
    assert result.returncode == 1
    assert "unsupported E2E SGLang image" in result.stderr
    assert calls == []
    assert not (tmp_path / "native.jsonl").exists()
    assert not (tmp_path / "github.env").exists()


@pytest.mark.parametrize("arguments", [("--unknown",), ("--verify-only", "one", "two")])
def test_invalid_mode_stops_before_tools(tmp_path, arguments):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=arguments,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert calls == [] and not (tmp_path / "native.jsonl").exists()


def test_verify_only_rejects_own_library_before_check_or_success_report(tmp_path):
    report = tmp_path / "native.json"
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=("--verify-only", str(report)),
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
            "IMAGE_DLPACK_LIB": "fixture-dlpack-library",
        },
    )
    assert result.returncode != 0
    assert "DLPack extension loaded its own library: 'fixture-dlpack-library'" in result.stderr
    assert calls == [] and not report.exists()
    assert not (tmp_path / "github.env").exists()


def test_ambient_dlpack_fixture_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_DLPACK_LIB", "ambient-must-not-be-used")
    result, _ = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={"IMAGE_REFLECTION_VERSION": "1.84.0"},
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "github.env").exists()


@pytest.mark.parametrize("state", ["valid", "own-library", "loaded-library", "replaced-api"])
def test_real_verifier_builtin_api_rejects_replacements(state):
    from types import SimpleNamespace

    verifier = load_cohort_verifier()
    api = object()
    torch = SimpleNamespace(Tensor=SimpleNamespace(__dlpack_c_exchange_api__=api))

    def load():
        if state == "replaced-api":
            torch.Tensor.__dlpack_c_exchange_api__ = object()
        return object() if state == "loaded-library" else None

    core = SimpleNamespace(
        _lib="fixture-library" if state == "own-library" else None,
        load_torch_c_dlpack_extension=load,
    )
    if state == "valid":
        verifier.verify_builtin_dlpack(core, torch, api)
    else:
        diagnostic = {
            "own-library": "loaded its own library: 'fixture-library'",
            "loaded-library": "loaded a replacement builtin API",
            "replaced-api": "replaced the Torch builtin C API",
        }[state]
        with pytest.raises(RuntimeError, match=diagnostic):
            verifier.verify_builtin_dlpack(core, torch, api)


def test_verifier_unexpected_failure_keeps_json_exit_and_traceback(tmp_path, monkeypatch, capsys):
    verifier = load_cohort_verifier()
    report = tmp_path / "unexpected.json"

    def broken(_):
        raise ValueError("fixture unexpected verifier failure")

    monkeypatch.setattr(verifier, "verify", broken)
    monkeypatch.setattr(sys, "argv", ["ci_verify_sglang_rocm_cohort.py", str(report)])
    assert verifier.main() == 1
    captured = capsys.readouterr()
    data = json.loads(report.read_text())
    assert json.loads(captured.out) == data
    assert data["verified"] is False
    assert "excluding generated bytecode bodies" in data["scope"]
    assert "Verifier imports bypass installed bytecode caches" in data["scope"]
    assert "later engine/pytest imports are not covered" in data["scope"]
    assert data["error"] == {"type": "ValueError", "message": "fixture unexpected verifier failure"}
    assert "Traceback (most recent call last)" in captured.err
    assert "ValueError: fixture unexpected verifier failure" in captured.err


def test_python_fixture_rejects_unknown_pip_commands(tmp_path):
    result, _ = run_installer(tmp_path, {})
    assert result.returncode == 0, result.stderr
    env = {
        **os.environ,
        "TOOL_LOG": str(tmp_path / "unexpected-pip.jsonl"),
        "TOOL_ENV_LOG": str(tmp_path / "unexpected-env.jsonl"),
    }
    unknown = subprocess.run(
        [str(tmp_path / "bin/python3"), "-m", "pip", "install", "unapproved-package"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert unknown.returncode != 0
    assert "unknown fixture pip command" in unknown.stderr
    assert not (tmp_path / "unexpected-pip.jsonl").exists()


def test_real_verifier_native_version_mismatch_reports_observed_and_expected(tmp_path, monkeypatch):
    verifier = load_cohort_verifier()
    constraint = tmp_path / "torch-rocm.txt"
    constraint.write_bytes(b"fixture unchanged constraint")
    monkeypatch.setattr(verifier, "CONSTRAINT", constraint)
    monkeypatch.setattr(
        verifier, "CONSTRAINT_SHA", hashlib.sha256(constraint.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(verifier.sys, "prefix", "/opt/venv")
    monkeypatch.setenv("SMG_SGLANG_ROCM_IMAGE", verifier.IMAGE)
    monkeypatch.setenv("PIP_CONSTRAINT", str(constraint))
    observed = {
        "sglang": "0.5.21",
        "torch": "2.11.0+rocm10.0.0",
        "apache-tvm-ffi": "0.2.0",
        "tilelang": "0.1.16",
        "torch-c-dlpack-ext": "0.1.6",
    }
    monkeypatch.setattr(verifier.md, "version", observed.__getitem__)

    # The real version guard must fail before distribution traversal/native imports.
    def unexpected_distributions():
        raise AssertionError("version mismatch reached installed/native inspection")

    monkeypatch.setattr(verifier.md, "distributions", unexpected_distributions)
    with pytest.raises(RuntimeError) as failure:
        verifier.verify({})
    message = str(failure.value)
    assert "ERROR: native dependency versions (TVM, TileLang, DLPack):" in message
    assert "observed=('0.2.0', '0.1.16', '0.1.6')" in message
    assert "expected=('0.1.12', '0.1.15', '0.1.5')" in message
    assert "full cohort observed=" in message and "expected=" in message


@pytest.mark.parametrize(
    "package,bad_version,expected_version",
    [("sglang", "0.5.22", "0.5.21"), ("torch", "2.12.0+rocm10.0.0", "2.11.0+rocm10.0.0")],
)
def test_real_verifier_reports_changed_sglang_or_torch_before_native_context(
    tmp_path, monkeypatch, package, bad_version, expected_version
):
    verifier = load_cohort_verifier()
    constraint = tmp_path / "torch-rocm.txt"
    constraint.write_bytes(b"fixture unchanged constraint")
    monkeypatch.setattr(verifier, "CONSTRAINT", constraint)
    monkeypatch.setattr(
        verifier, "CONSTRAINT_SHA", hashlib.sha256(constraint.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(verifier.sys, "prefix", "/opt/venv")
    monkeypatch.setenv("SMG_SGLANG_ROCM_IMAGE", verifier.IMAGE)
    monkeypatch.setenv("PIP_CONSTRAINT", str(constraint))
    versions = {
        "sglang": "0.5.21",
        "torch": "2.11.0+rocm10.0.0",
        "apache-tvm-ffi": "0.1.12",
        "tilelang": "0.1.15",
        "torch-c-dlpack-ext": "0.1.5",
    }
    versions[package] = bad_version
    monkeypatch.setattr(verifier.md, "version", versions.__getitem__)

    def unexpected_distributions():
        raise AssertionError("version mismatch reached installed/native inspection")

    monkeypatch.setattr(verifier.md, "distributions", unexpected_distributions)
    with pytest.raises(RuntimeError) as failure:
        verifier.verify({})
    message = str(failure.value)
    leading_diagnostic = message.split(";", 1)[0]
    assert repr(package) in leading_diagnostic
    assert repr(bad_version) in leading_diagnostic and repr(expected_version) in leading_diagnostic
    assert "apache-tvm-ffi" not in leading_diagnostic
    assert "tilelang" not in leading_diagnostic and "torch-c-dlpack-ext" not in leading_diagnostic
    assert "native dependency versions (TVM, TileLang, DLPack)" not in message
    assert "full cohort observed=" in message and "expected=" in message


# Both benchmark job templates install more packages after their initial image
# setup. Check that the strict cohort verification also runs after that install.
# These tests inspect Source only: no backend, model or native verifier starts.
def _nightly_sglang_postinstall_contract(job):
    import shlex

    steps = job["steps"]
    checks = [
        (index, entry)
        for index, entry in enumerate(steps)
        if "ci_setup_sglang_rocm_image.sh --verify-only" in entry.get("run", "")
    ]
    assert len(checks) == 1, "expected one post-install SGLang cohort check"
    index, check = checks[0]
    installs = [
        position
        for position, entry in enumerate(steps)
        if "pip install wheel/*.whl" in entry.get("run", "")
        and "bash scripts/ci_install_e2e_deps.sh" in entry.get("run", "")
    ]
    benchmarks = [
        position
        for position, entry in enumerate(steps)
        if "pytest e2e_test/benchmarks/test_nightly_perf.py" in entry.get("run", "")
    ]
    assert len(installs) == len(benchmarks) == 1
    assert index == installs[0] + 1 and index < benchmarks[0]
    assert check["if"] == (
        "steps.filter.outputs.skip != 'true' && matrix.variant.runtime == 'sglang'"
    )
    assert check["env"] == {"SMG_SGLANG_ROCM_IMAGE": "${{ matrix.variant.image }}"}
    assert check["timeout-minutes"] == 2
    assert check.get("continue-on-error", False) is False
    commands = [shlex.split(line) for line in check["run"].splitlines() if line.strip()]
    assert commands == [
        ["mkdir", "-p", "nightly_gateway_logs"],
        [
            "bash",
            "scripts/ci_setup_sglang_rocm_image.sh",
            "--verify-only",
            "nightly_gateway_logs/sglang-native-environment-prepared.json",
        ],
    ]
    return index


def _nightly_sglang_benchmark_job(job_id):
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/nightly-benchmark-amd.yml").read_text())
    return workflow["jobs"][job_id]


@pytest.mark.parametrize("job_id", ["single-worker", "multi-worker"])
def test_nightly_sglang_postinstall_after_dependencies(job_id):
    _nightly_sglang_postinstall_contract(_nightly_sglang_benchmark_job(job_id))


@pytest.mark.parametrize("job_id", ["single-worker", "multi-worker"])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "before_dependencies",
        "unfiltered",
        "wrong_runtime",
        "wrong_image",
        "continue_on_error",
        "suppressed_exit",
        "wrong_report",
        "duplicate",
        "unbounded_timeout",
    ],
)
def test_nightly_sglang_postinstall_rejects_broken_wiring(job_id, mutation):
    import copy

    job = copy.deepcopy(_nightly_sglang_benchmark_job(job_id))
    index = _nightly_sglang_postinstall_contract(job)
    check = job["steps"][index]
    if mutation == "missing":
        job["steps"].pop(index)
    elif mutation == "before_dependencies":
        job["steps"].insert(index - 1, job["steps"].pop(index))
    elif mutation == "unfiltered":
        check["if"] = "matrix.variant.runtime == 'sglang'"
    elif mutation == "wrong_runtime":
        check["if"] = "steps.filter.outputs.skip != 'true' && matrix.variant.runtime == 'vllm'"
    elif mutation == "wrong_image":
        check["env"]["SMG_SGLANG_ROCM_IMAGE"] = "an-unbound-image"
    elif mutation == "continue_on_error":
        check["continue-on-error"] = True
    elif mutation == "suppressed_exit":
        check["run"] += " || true"
    elif mutation == "wrong_report":
        check["run"] = check["run"].replace(
            "nightly_gateway_logs/sglang-native-environment-prepared.json", "unretained.json"
        )
    elif mutation == "duplicate":
        job["steps"].insert(index, copy.deepcopy(check))
    elif mutation == "unbounded_timeout":
        check["timeout-minutes"] = 1440
    with pytest.raises(AssertionError):
        _nightly_sglang_postinstall_contract(job)


# The verifier owns the image pin. Exercise real helper parsing with private
# Source fixtures; these commands never install packages or load native modules.
def _pin_source_fixture(tmp_path, monkeypatch, declaration, *, missing_file=False):
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    for name in ("ci_setup_sglang_rocm_image.sh", "ci_retry.sh"):
        original = ROOT / "scripts" / name
        (scripts / name).write_bytes(original.read_bytes())
    verifier = (ROOT / "scripts/ci_verify_sglang_rocm_cohort.py").read_text()
    original_pin = 'IMAGE = "' + FIXED_IMAGE + '"'
    assert verifier.splitlines().count(original_pin) == 1
    if not missing_file:
        (scripts / "ci_verify_sglang_rocm_cohort.py").write_text(
            verifier.replace(original_pin, declaration, 1)
        )
    monkeypatch.setattr(sys.modules[__name__], "ROOT", source)
    return source


def _pin_rejects_before_tools(tmp_path, monkeypatch):
    import shlex

    # If the guard reaches Python or the old dirname subprocess, fail visibly.
    marker = tmp_path / "unexpected-tool"
    monkeypatch.setattr(
        sys.modules[__name__],
        "FAKE",
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\nraise SystemExit(99)\n",
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("dirname", "pin_payload"):
        command = binary / name
        command.write_text(
            "#!/bin/sh\nprintf unexpected > " + shlex.quote(str(marker)) + "\nexit 99\n"
        )
        command.chmod(0o755)
    return marker


@pytest.mark.parametrize("requested", ["new-pin", "old-pin"])
def test_verify_only_follows_the_verifier_image_pin(tmp_path, monkeypatch, requested):
    new_image = FIXED_IMAGE[:-64] + "a" * 64
    _pin_source_fixture(tmp_path, monkeypatch, 'IMAGE = "' + new_image + '"')
    if requested == "old-pin":
        marker = _pin_rejects_before_tools(tmp_path, monkeypatch)
    report = tmp_path / "native.json"
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=("--verify-only", str(report)),
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "SMG_SGLANG_ROCM_IMAGE": new_image if requested == "new-pin" else FIXED_IMAGE,
        },
    )
    if requested == "new-pin":
        assert result.returncode == 0, result.stderr
        assert calls == [["-m", "pip", "check"]]
        assert json.loads(report.read_text())["verified"] is True
        assert not (tmp_path / "github.env").exists()
    else:
        assert result.returncode == 1, result.stderr
        assert "unsupported E2E SGLang image" in result.stderr
        assert calls == [] and not marker.exists()
        assert not report.exists() and not (tmp_path / "native.jsonl").exists()
        assert not (tmp_path / "github.env").exists()


@pytest.mark.parametrize(
    "state",
    [
        "missing-file",
        "missing-pin",
        "duplicate",
        "unclosed",
        "short-digest",
        "computed",
        "multiline",
        "annotated",
        "source-looking",
    ],
)
def test_verify_only_rejects_invalid_authoritative_pin_before_tools(tmp_path, monkeypatch, state):
    literal = 'IMAGE = "' + FIXED_IMAGE + '"'
    declaration = {
        "missing-file": literal,
        "missing-pin": "# no IMAGE declaration",
        "duplicate": literal + "\n" + literal,
        "unclosed": 'IMAGE = "' + FIXED_IMAGE,
        "short-digest": 'IMAGE = "' + FIXED_IMAGE[:-1] + '"',
        "computed": literal + ' + ""',
        "multiline": 'IMAGE = (\n    "' + FIXED_IMAGE + '"\n)',
        "annotated": literal.replace("IMAGE =", "IMAGE: str ="),
        "source-looking": 'IMAGE = "$(pin_payload)"',
    }[state]
    _pin_source_fixture(tmp_path, monkeypatch, declaration, missing_file=state == "missing-file")
    marker = _pin_rejects_before_tools(tmp_path, monkeypatch)
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=("--verify-only",),
        extra_env={"SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE},
    )
    assert result.returncode == 1, result.stderr
    assert "invalid verifier IMAGE pin" in result.stderr
    assert calls == [] and not marker.exists()
    assert not (tmp_path / "native.jsonl").exists()
    assert not (tmp_path / "github.env").exists()


def test_verify_only_accepts_a_single_quoted_authoritative_pin(tmp_path, monkeypatch):
    _pin_source_fixture(tmp_path, monkeypatch, "IMAGE = '" + FIXED_IMAGE + "'")
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=("--verify-only",),
        extra_env={"IMAGE_REFLECTION_VERSION": "1.84.0", "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE},
    )
    assert result.returncode == 0, result.stderr
    assert calls == [["-m", "pip", "check"]]
    assert not (tmp_path / "github.env").exists()


def _load_cache_fixture(source):
    spec = importlib.util.spec_from_file_location("smg_cohort_cache_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["timestamp", "checked-hash", "unchecked-hash"])
def test_verifier_imports_ignore_header_valid_recorded_cache(tmp_path, monkeypatch, mode):
    import marshal

    verifier = load_cohort_verifier()
    monkeypatch.setattr(sys, "pycache_prefix", None)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    source = tmp_path / "payload.py"
    trusted = b'VALUE = "source"\n'
    source.write_bytes(trusted)
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir()
    if mode == "timestamp":
        stat = source.stat()
        header = (
            b"\0" * 4
            + (int(stat.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
            + (len(trusted) & 0xFFFFFFFF).to_bytes(4, "little")
        )
    else:
        flags = 3 if mode == "checked-hash" else 1
        header = flags.to_bytes(4, "little") + importlib.util.source_hash(trusted)
    # Only this test-authored code is serialized. No cached bytes are unmarshaled
    # by the verifier; a matching source header does not authenticate this body.
    poison = compile('VALUE = "cached"\n', str(source), "exec")
    cache.write_bytes(importlib.util.MAGIC_NUMBER + header + marshal.dumps(poison))
    folder = tmp_path / "fixture.dist-info"
    folder.mkdir()
    encoded = base64.urlsafe_b64encode(hashlib.sha256(trusted).digest()).rstrip(b"=").decode()
    record = folder / "RECORD"
    record.write_text(
        f"payload.py,sha256={encoded},{len(trusted)}\n"
        f"{cache.relative_to(tmp_path)},,\nfixture.dist-info/RECORD,,\n"
    )
    assert verifier.verify_record(tmp_path, folder) == {
        "payload.py",
        str(cache.relative_to(tmp_path)),
        "fixture.dist-info/RECORD",
    }
    assert _load_cache_fixture(source).VALUE == "cached", "fixture must reproduce the original gap"
    before = {path: path.read_bytes() for path in (source, cache, record)}
    with verifier._source_imports() as environment:
        prefix = Path(environment["PYTHONPYCACHEPREFIX"])
        assert sys.pycache_prefix == str(prefix)
        assert sys.dont_write_bytecode is True
        assert _load_cache_fixture(source).VALUE == "source"
        assert list(prefix.rglob("*")) == []
    assert not prefix.exists()
    assert sys.pycache_prefix is None and sys.dont_write_bytecode is True
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("failure", [False, True])
def test_source_import_policy_restores_settings_and_outer_environment(
    tmp_path, monkeypatch, failure
):
    verifier = load_cohort_verifier()
    previous_prefix = str(tmp_path / "outer-cache")
    monkeypatch.setattr(sys, "pycache_prefix", previous_prefix)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", previous_prefix)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "outer-setting")
    before = dict(os.environ)
    prefixes = []

    def exercise():
        with verifier._source_imports() as environment:
            prefix = Path(environment["PYTHONPYCACHEPREFIX"])
            prefixes.append(prefix)
            assert prefix.is_dir() and str(prefix) != previous_prefix
            assert sys.pycache_prefix == str(prefix) and sys.dont_write_bytecode is True
            assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
            assert os.environ == before
            if failure:
                raise ValueError("fixture import failed")

    if failure:
        with pytest.raises(ValueError, match="fixture import failed"):
            exercise()
    else:
        exercise()
    assert len(prefixes) == 1 and not prefixes[0].exists()
    assert sys.pycache_prefix == previous_prefix and sys.dont_write_bytecode is False
    assert os.environ == before


def _fixed_cohort_before_native(tmp_path, monkeypatch, *, pip_status=0):
    """Reach real verify guards with closed filesystem/native/command fixtures."""
    from types import SimpleNamespace

    verifier = load_cohort_verifier()
    monkeypatch.setattr(sys, "pycache_prefix", None)
    venv = tmp_path / "venv"
    root = venv / "lib/python3.12/site-packages"
    root.mkdir(parents=True)
    constraint = venv / "torch-rocm.txt"
    constraint.write_bytes(b"fixture fixed constraint\n")
    maps = tmp_path / "maps"
    maps.write_text("")
    # Relocate fixed absolute paths into the fixture, retaining the verifier's
    # real containment, RECORD, payload and anchor checks.
    original_path = verifier.Path

    def fixture_path(value):
        return {
            "/opt/venv": venv,
            "/opt/venv/bin": venv / "bin",
            "/proc/self/maps": maps,
        }.get(str(value), original_path(value))

    monkeypatch.setattr(verifier, "Path", fixture_path)
    monkeypatch.setattr(verifier, "CONSTRAINT", constraint)
    monkeypatch.setattr(
        verifier, "CONSTRAINT_SHA", hashlib.sha256(constraint.read_bytes()).hexdigest()
    )
    monkeypatch.setenv("PIP_CONSTRAINT", str(constraint))
    monkeypatch.setenv("SMG_SGLANG_ROCM_IMAGE", verifier.IMAGE)
    # A synthetic interpreter context keeps these direct-verify controls
    # independent of unrelated modules imported by the surrounding CPU suite.
    interpreter = SimpleNamespace(
        prefix="/opt/venv",
        executable=sys.executable,
        modules={},
        pycache_prefix=None,
        dont_write_bytecode=False,
    )
    monkeypatch.setattr(verifier, "sys", interpreter)
    monkeypatch.setattr(verifier, "sysconfig", SimpleNamespace(get_path=lambda name: str(root)))
    versions = {
        "sglang": "0.5.21",
        "torch": "2.11.0+rocm10.0.0",
        "apache-tvm-ffi": "0.1.12",
        "tilelang": "0.1.15",
        "torch-c-dlpack-ext": "0.1.5",
    }
    for name in verifier.ANCHORS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("fixture anchor: " + name + "\n").encode())
    (root / "tvm_ffi/__init__.py").write_text("# fixture TVM package\n")
    extension = root / "torch_c_dlpack_ext-0.1.5.dist-info"
    (extension / "WHEEL").write_text(
        "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    (extension / "INSTALLER").write_text("pip\n")
    (extension / "REQUESTED").write_text("")
    (extension / "direct_url.json").write_text(
        json.dumps(
            {
                "url": verifier.SDIST_URL,
                "archive_info": {"hashes": {"sha256": verifier.SDIST_SHA}},
            }
        )
    )
    anchors = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in verifier.ANCHORS
    }
    monkeypatch.setattr(verifier, "ANCHORS", anchors)
    distributions = []
    for name, version, package in (
        ("apache-tvm-ffi", "0.1.12", "tvm_ffi"),
        ("tilelang", "0.1.15", "tilelang"),
        ("torch-c-dlpack-ext", "0.1.5", "torch_c_dlpack_ext"),
    ):
        folder = root / (name.replace("-", "_") + "-" + version + ".dist-info")
        files = sorted(
            path
            for parent in (root / package, folder)
            for path in parent.rglob("*")
            if path.is_file()
        )
        entries = [(str(path.relative_to(root)), path) for path in files]
        if name == "apache-tvm-ffi":
            (venv / "bin").mkdir()
            for console in ("tvm-ffi-config", "tvm-ffi-stubgen"):
                path = venv / "bin" / console
                path.write_text("# fixture console\n")
                entries.append(("../../../bin/" + console, path))
        elif name == "torch-c-dlpack-ext":
            entries.append(("build_backend.py", root / "build_backend.py"))
        rows = []
        for relative, path in entries:
            body = path.read_bytes()
            encoded = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
            rows.append(f"{relative},sha256={encoded},{len(body)}\n")
        (folder / "RECORD").write_text("".join(rows) + folder.name + "/RECORD,,\n")
        distributions.append(
            SimpleNamespace(metadata={"Name": name}, _path=folder, locate_file=lambda _: root)
        )
    monkeypatch.setattr(
        verifier,
        "md",
        SimpleNamespace(version=versions.__getitem__, distributions=lambda: distributions),
    )
    native_calls, pip_calls = [], []

    class CpuTensor:
        def __init__(self, values=None):
            self.values = [[0, 2], [4, 6], [8, 10]] if values is None else values
            self.device = SimpleNamespace(type="cpu")
            self.dtype, self.shape = "fixture-int64", (3, 2)

        def reshape(self, *shape):
            assert shape == (3, 4)
            return self

        def __getitem__(self, key):
            if isinstance(key[0], slice):
                assert key == (slice(None), slice(None, None, 2))
                return self
            return SimpleNamespace(item=lambda: self.values[key[0]][key[1]])

        def __setitem__(self, key, value):
            self.values[key[0]][key[1]] = value

        def data_ptr(self):
            return id(self.values)

        def stride(self):
            return (4, 2)

        def tolist(self):
            return [list(row) for row in self.values]

    api = type("PyCapsule", (), {})()
    tensor_type = type("Tensor", (), {"__dlpack_c_exchange_api__": api})
    torch = SimpleNamespace(
        __version__=versions["torch"],
        version=SimpleNamespace(hip="fixture-rocm"),
        Tensor=tensor_type,
        arange=lambda count, device: CpuTensor(),
        utils=SimpleNamespace(
            dlpack=SimpleNamespace(from_dlpack=lambda source: CpuTensor(source.values))
        ),
    )

    def native_module(name):
        native_calls.append(name)
        if name == "torch":
            return torch
        relative = "torch_c_dlpack_ext/core.py" if name.endswith(".core") else name + "/__init__.py"
        return SimpleNamespace(
            __file__=str(root / relative), _lib=None, load_torch_c_dlpack_extension=lambda: None
        )

    monkeypatch.setattr(
        verifier,
        "importlib",
        SimpleNamespace(
            import_module=native_module,
            util=SimpleNamespace(
                cache_from_source=importlib.util.cache_from_source,
                source_from_cache=importlib.util.source_from_cache,
                find_spec=lambda name: SimpleNamespace(origin=str(root / "build_backend.py")),
            ),
        ),
    )

    def pip(command, **kwargs):
        assert command == [interpreter.executable, "-m", "pip", "check"]
        prefix = Path(kwargs["env"]["PYTHONPYCACHEPREFIX"])
        assert prefix.is_dir() and list(prefix.iterdir()) == []
        assert interpreter.pycache_prefix == str(prefix) and interpreter.dont_write_bytecode is True
        pip_calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=pip_status, stdout="fixture pip output\n", stderr="fixture pip error\n"
        )

    monkeypatch.setattr(verifier, "subprocess", SimpleNamespace(run=pip))
    return verifier, root, native_calls, pip_calls


def test_direct_verify_rejects_anchor_mismatch_before_imports(tmp_path, monkeypatch):
    verifier, root, native_calls, pip_calls = _fixed_cohort_before_native(tmp_path, monkeypatch)
    # Change the official expected anchor only: every installed RECORD hash
    # remains valid, so this must reach the actual anchor comparison.
    verifier.ANCHORS["tilelang/__init__.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="Official payload anchor differs: tilelang/__init__.py"):
        verifier.verify({})
    assert (root / "tilelang/__init__.py").is_file()
    assert native_calls == [] and pip_calls == []


@pytest.mark.parametrize("name", ["torch", "tvm_ffi.core", "torch_c_dlpack_ext"])
def test_direct_verify_rejects_preloaded_protected_modules(tmp_path, monkeypatch, name):
    verifier, _, native_calls, pip_calls = _fixed_cohort_before_native(tmp_path, monkeypatch)
    preloaded = object()
    verifier.sys.modules[name] = preloaded
    with pytest.raises(
        RuntimeError, match="Protected modules loaded before source-only verification"
    ):
        verifier.verify({})
    assert verifier.sys.modules[name] is preloaded
    assert verifier.sys.pycache_prefix is None and verifier.sys.dont_write_bytecode is False
    assert native_calls == [] and pip_calls == []


@pytest.mark.parametrize("pip_status", [0, 23])
def test_direct_verify_checks_pip_after_real_filesystem_and_cpu_guards(
    tmp_path, monkeypatch, pip_status
):
    verifier, root, native_calls, pip_calls = _fixed_cohort_before_native(
        tmp_path, monkeypatch, pip_status=pip_status
    )
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    outer_environment = dict(os.environ)
    report = {"verified": False}
    if pip_status:
        with pytest.raises(RuntimeError, match="pip check failed"):
            verifier.verify(report)
    else:
        verifier.verify(report)
    assert len(pip_calls) == 1
    _, options = pip_calls[0]
    assert options["capture_output"] is True
    assert options["text"] is True and options["timeout"] == 60
    assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not Path(options["env"]["PYTHONPYCACHEPREFIX"]).exists()
    assert options["env"] is not os.environ and os.environ == outer_environment
    assert verifier.sys.pycache_prefix is None and verifier.sys.dont_write_bytecode is False
    assert report["pip_check"] == {
        "returncode": pip_status,
        "stdout": "fixture pip output\n",
        "stderr": "fixture pip error\n",
    }
    assert report["official_anchors"] == len(verifier.ANCHORS)
    assert report["verified"] is False, "only the CLI may set success after verify returns"
    assert native_calls == [
        "torch",
        "sglang",
        "mooncake",
        "smg_grpc_servicer",
        "tvm_ffi",
        "tilelang",
        "xgrammar",
        "torch_c_dlpack_ext.core",
    ]
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("replacement", ["own-library", "builtin-api"])
def test_direct_verify_rejects_dlpack_replacements_before_pip(tmp_path, monkeypatch, replacement):
    verifier, _, native_calls, pip_calls = _fixed_cohort_before_native(tmp_path, monkeypatch)
    original_import = verifier.importlib.import_module
    imported = {}

    def import_module(name):
        module = original_import(name)
        imported[name] = module
        if name == "torch_c_dlpack_ext.core":
            if replacement == "own-library":
                module._lib = "fixture-owned-addon"
            else:
                imported["torch"].Tensor.__dlpack_c_exchange_api__ = object()
        return module

    monkeypatch.setattr(verifier.importlib, "import_module", import_module)
    message = (
        "DLPack extension loaded its own library"
        if replacement == "own-library"
        else "DLPack extension replaced the Torch builtin C API"
    )
    with pytest.raises(RuntimeError, match=message):
        verifier.verify({})
    assert native_calls[-1] == "torch_c_dlpack_ext.core"
    assert pip_calls == []
