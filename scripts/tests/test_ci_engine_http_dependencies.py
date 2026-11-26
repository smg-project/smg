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
FIXED_IMAGE = (
    "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
    "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
)
FAKE = r"""
import importlib.metadata
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

args = sys.argv[1:]
if args[0].endswith("ci_verify_sglang_rocm_cohort.py"):
    if args[1:] == ["--repair-plan"]:
        if os.environ.get("REPAIR_PLAN_ERROR"):
            raise SystemExit("Unsafe uninstall RECORD path (controlled admission failure)")
        packages = json.loads(os.environ.get("NATIVE_REPAIR_PACKAGES",
                              '["apache-tvm-ffi", "tilelang", "torch-c-dlpack-ext"]'))
        if os.environ.get("NATIVE_CORRUPTION_FILE"):
            packages = (["tilelang"] if Path(os.environ["NATIVE_CORRUPTION_FILE"]).read_text()
                        != "repaired" else [])
        print("\n".join(packages))
        raise SystemExit(0)
    with open(os.environ["TOOL_NATIVE_LOG"], "a") as log:
        log.write(json.dumps(args) + "\n")
    # Exercise the real CLI report/exit contract and real builtin guard.
    # Only the filesystem/native/pip inputs are software fixtures here.
    spec = importlib.util.spec_from_file_location("cohort_verifier", args[0])
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    def fixture_verify(report):
        if os.environ.get("FAIL_NATIVE_CHECK"):
            raise RuntimeError("injected native check failure")
        api = object()
        torch = types.SimpleNamespace(Tensor=types.SimpleNamespace(__dlpack_c_exchange_api__=api))
        core = types.SimpleNamespace(_lib=os.environ.get("IMAGE_DLPACK_LIB"),
                                     load_torch_c_dlpack_extension=lambda: None)
        verifier.verify_builtin_dlpack(core, torch, api)
        if os.environ.get("NATIVE_CORRUPTION_FILE"):
            verifier.require(Path(os.environ["NATIVE_CORRUPTION_FILE"]).read_text() == "repaired",
                             "Native official anchor still differs")
        check = ["-m", "pip", "check"]
        with open(os.environ["TOOL_LOG"], "a") as log:
            log.write(json.dumps(check) + "\n")
        with open(os.environ["TOOL_ENV_LOG"], "a") as log:
            log.write(json.dumps({key: os.environ.get(key) for key in (
                "PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT", "UV_CONSTRAINT"
            )}) + "\n")
        status = 43 if os.environ.get("FAIL_PIP_STAGE") == "check" else 0
        report["pip_check"] = {"returncode": status}
        verifier.require(status == 0, "pip check failed")

    verifier.verify = fixture_verify
    sys.argv = args
    raise SystemExit(verifier.main())
if args == ["-V"]:
    print("Python (test fixture)")
elif args[0] == "-c":
    def version(name):
        value = {
            "grpcio-reflection": os.environ.get("IMAGE_REFLECTION_VERSION"),
            "opentelemetry-api": os.environ.get("IMAGE_OTEL_API_VERSION", "1.45.0"),
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
        if os.environ.get("FAIL_TORCH_PREFLIGHT"):
            raise ImportError("injected installed Torch cache failure")
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
        for name in ("sglang", "mooncake", "smg_grpc_servicer"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["sglang"].__version__ = version("sglang")
    exec(args[1])
elif args[:2] == ["-m", "pip"]:
    known = (
        args == ["-m", "pip", "check"]
        or args == ["-m", "pip", "install", "clients/python/"]
        or (args[:4] == ["-m", "pip", "install", "e2e_test/"]
            and all(arg.startswith(("mcp", "fastapi", "starlette", "opentelemetry-api==")) for arg in args[4:]))
        or (args[:3] == ["-m", "pip", "install"] and len(args) == 5
            and args[3].startswith("grpcio-health-checking==")
            and args[4].startswith("grpcio-reflection=="))
        or args == ["-m", "pip", "install", "-e", "crates/grpc_client/python/", "-e", "grpc_servicer/"]
        or (args[:8] == ["-m", "pip", "install", "--no-cache-dir", "--no-build-isolation",
                        "--no-binary=torch-c-dlpack-ext", "--force-reinstall", "--no-deps"]
            and args[8:] and len(args[8:]) == len(set(args[8:]))
            and all(arg in {"apache-tvm-ffi==0.1.12", "tilelang==0.1.15",
                            "torch-c-dlpack-ext @ https://files.pythonhosted.org/packages/37/de/921b6491efce5c389a5ef9bbed3d2d6660005840dae488124173180859ab/torch_c_dlpack_ext-0.1.5.tar.gz#sha256=d06f0357d575d22a168cc77acb9020fc4bae30968ceb6718a055dcbe92bacabe"}
                    for arg in args[8:]))
    )
    if not known:
        raise RuntimeError("unknown fixture pip command: " + repr(args))
    if (os.environ.get("NATIVE_CORRUPTION_FILE") and "tilelang==0.1.15" in args
            and "--force-reinstall" in args):
        Path(os.environ["NATIVE_CORRUPTION_FILE"]).write_text("repaired")
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


@pytest.fixture(autouse=True)
def ordinary_core_consumer(monkeypatch):
    # Ordinary core tests must not inherit a job's optional consumer mode.
    # The unsupported-consumer negative explicitly enables it again itself.
    monkeypatch.setenv("E2E_SGLANG_GPTOSS_MXFP4_BF16", "0")


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
        **{
            key: os.environ[key]
            for key in ("LD_LIBRARY_PATH", "PYTHONPYCACHEPREFIX")
            if key in os.environ
        },
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "E2E_SGLANG_GPTOSS_MXFP4_BF16": "0",
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
        "SMG_CI_OTEL_API_VERSION",
        "IMAGE_OTEL_API_VERSION",
        "SMG_SGLANG_ROCM_IMAGE",
        "IMAGE_REFLECTION_VERSION",
        "IMAGE_SGLANG_VERSION",
        "IMAGE_TORCH_VERSION",
        "IMAGE_TORCH_HIP",
        "IMAGE_TORCH_C_API",
        "IMAGE_DLPACK_LIB",
        "FAIL_PIP_STAGE",
        "FAIL_NATIVE_CHECK",
        "FAIL_TORCH_PREFLIGHT",
        "NATIVE_REPAIR_PACKAGES",
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


@pytest.mark.parametrize("native_repair", [False, True])
@pytest.mark.parametrize("image_version", ["1.84.0", "1.90.0"])
def test_sglang_image_companion_survives_separate_actions_steps(
    tmp_path, image_version, native_repair
):
    # Metadata/native modules/pip are fixtures here; this proves only shell wiring.
    github_env = tmp_path / "github.env"
    github_env.write_text("EXISTING=value\n")
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={
            "IMAGE_REFLECTION_VERSION": image_version,
            **({"SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE} if native_repair else {}),
        },
    )
    assert result.returncode == 0, result.stderr
    expected_native = [
        [
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-build-isolation",
            "--no-binary=torch-c-dlpack-ext",
            "--force-reinstall",
            "--no-deps",
            "apache-tvm-ffi==0.1.12",
            "tilelang==0.1.15",
            EXTENSION_SOURCE,
        ],
        ["-m", "pip", "install", "-e", "crates/grpc_client/python/", "-e", "grpc_servicer/"],
        ["-m", "pip", "check"],
    ]
    assert calls == (expected_native if native_repair else [expected_native[1]])
    assert github_env.read_text() == (
        "EXISTING=value\n"
        + ("SMG_CI_OTEL_API_VERSION=1.45.0\n" if native_repair else "")
        + f"SMG_CI_GRPC_COMPANION_VERSION={image_version}\n"
    )
    if native_repair:
        native = [json.loads(line) for line in (tmp_path / "native.jsonl").read_text().splitlines()]
        assert native == [[str(ROOT / "scripts/ci_verify_sglang_rocm_cohort.py")]]
    else:
        assert not (tmp_path / "native.jsonl").exists()
    handoff = dict(line.split("=", 1) for line in github_env.read_text().splitlines())
    (tmp_path / "pip.jsonl").unlink()
    # A new bash process consumes the real producer output, not the setup shell.
    result, calls = run_installer(tmp_path, {}, extra_env=handoff)
    assert calls[0] == [
        "-m",
        "pip",
        "install",
        "e2e_test/",
        *(["opentelemetry-api==1.45.0"] if native_repair else []),
    ]
    assert result.returncode == 0, result.stderr
    assert calls[-1] == [
        "-m",
        "pip",
        "install",
        "grpcio-health-checking==" + image_version,
        "grpcio-reflection==" + image_version,
    ]


@pytest.mark.parametrize("hip", ["legacy-HIP", ""])
def test_unpinned_legacy_setup_does_not_require_native_verifier(tmp_path, hip):
    # An existing caller with another ROCm cohort need not ship the new verifier.
    folder = tmp_path / "legacy"
    folder.mkdir()
    for name in ("ci_setup_sglang_rocm_image.sh", "ci_retry.sh"):
        (folder / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    result, calls = run_installer(
        tmp_path,
        {},
        script=str(folder / "ci_setup_sglang_rocm_image.sh"),
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "IMAGE_SGLANG_VERSION": "0.4.9",
            "IMAGE_OTEL_API_VERSION": "",
            "IMAGE_TORCH_VERSION": "2.8.0+rocm6.4",
            "IMAGE_TORCH_C_API": "absent",
            "IMAGE_TORCH_HIP": hip,
        },
    )
    assert calls == [
        ["-m", "pip", "install", "-e", "crates/grpc_client/python/", "-e", "grpc_servicer/"]
    ]
    assert (result.returncode == 0) == bool(hip), result.stderr
    assert (tmp_path / "github.env").exists() == bool(hip)
    assert not (tmp_path / "native.jsonl").exists()


def test_explicit_unsupported_repair_image_stops_before_tools(tmp_path):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={"SMG_SGLANG_ROCM_IMAGE": "unsupported", "IMAGE_REFLECTION_VERSION": "1.84.0"},
    )
    assert result.returncode == 1
    assert "unsupported E2E SGLang image" in result.stderr
    assert calls == [] and not (tmp_path / "github.env").exists()
    assert not (tmp_path / "native.jsonl").exists()


@pytest.mark.parametrize("version", ["1.45.0", "1.46.0"])
def test_sglang_otel_requirement_joins_the_e2e_resolver(tmp_path, version):
    result, calls = run_installer(
        tmp_path,
        {"sglang": ["fastapi>=0.120.1"]},
        extra_env={"SMG_CI_OTEL_API_VERSION": version},
    )
    assert result.returncode == 0, result.stderr
    assert calls[0] == [
        "-m",
        "pip",
        "install",
        "e2e_test/",
        "fastapi>=0.120.1",
        "opentelemetry-api==" + version,
    ]
    # No separate post-install API downgrade can leave FastMCP inconsistent.
    assert all("opentelemetry-api==" + version not in call for call in calls[1:])


def test_sglang_invalid_otel_handoff_stops_before_pip(tmp_path):
    result, calls = run_installer(
        tmp_path, {}, extra_env={"SMG_CI_OTEL_API_VERSION": "1.45.0\nother-package"}
    )
    assert result.returncode != 0 and "InvalidVersion" in result.stderr
    assert calls == []


@pytest.mark.parametrize("version", ["", "not-a-version"])
def test_sglang_invalid_image_otel_metadata_stops_before_install(tmp_path, version):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
            "IMAGE_OTEL_API_VERSION": version,
        },
    )
    assert result.returncode != 0 and "InvalidVersion" in result.stderr
    assert calls == []
    assert not (tmp_path / "github.env").exists()


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
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
            **bad_image,
        },
    )
    assert result.returncode != 0
    assert "ERROR:" in result.stderr
    assert diagnostic in result.stderr
    assert calls == [] and not (tmp_path / "github.env").exists()


@pytest.mark.parametrize("stage,exitcode", [("cohort", 42), ("native", 1), ("check", 1)])
def test_rocm_repair_failure_does_not_publish_handoff(tmp_path, stage, exitcode):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
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
        extra_env={
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
            **expected,
        },
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
    assert native == [[str(ROOT / "scripts/ci_verify_sglang_rocm_cohort.py"), "--", str(report)]]


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
    assert result.returncode == 1
    failed = json.loads((tmp_path / "native.json").read_text())
    assert failed["verified"] is False and failed["pip_check"]["returncode"] == 43
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
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
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
    assert calls == []
    failed = json.loads(report.read_text())
    assert failed["verified"] is False and "own library" in failed["error"]["message"]
    assert not (tmp_path / "github.env").exists()


def test_ambient_dlpack_fixture_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_DLPACK_LIB", "ambient-must-not-be-used")
    result, _ = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={"IMAGE_REFLECTION_VERSION": "1.84.0", "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE},
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
    backend_cache = root / "__pycache__" / f"build_backend.{sys.implementation.cache_tag}.pyc"
    backend_cache.parent.mkdir()
    backend_cache.write_bytes(b"inert fixture cache; verifier never executes cache bodies")
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
        if name == "torch-c-dlpack-ext":
            rows.append(str(backend_cache.relative_to(root)) + ",,\n")
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


def test_core_rejects_mxfp4_extension_before_native_imports(tmp_path, monkeypatch, capsys):
    verifier = load_cohort_verifier()
    monkeypatch.setenv("E2E_SGLANG_GPTOSS_MXFP4_BF16", "1")

    def unexpected_import(*args, **kwargs):
        raise AssertionError("native import before the unsupported-consumer guard")

    monkeypatch.setattr(verifier.importlib, "import_module", unexpected_import)
    report = tmp_path / "unsupported-extension.json"
    monkeypatch.setattr(sys, "argv", ["ci_verify_sglang_rocm_cohort.py", str(report)])
    assert verifier.main() == 1
    data = json.loads(report.read_text())
    assert data["verified"] is False
    assert data["error"]["type"] == "RuntimeError"
    assert "MXFP4 consumer requires its separately qualified extension" in data["error"]["message"]
    assert "gptoss_mxfp4_sdk" not in data
    assert "Traceback" in capsys.readouterr().err


VLLM_TOOL_FIXTURE = r"""
import builtins
import json
import os
import sys
import types
from pathlib import Path

args = sys.argv[1:]
role = Path(sys.argv[0]).name

def record(event):
    with open(os.environ["VLLM_TOOL_LOG"], "a") as stream:
        stream.write(json.dumps(event) + "\n")

record({"tool": role, "argv": args})
if role == "pip":
    assert args in (
        ["install", "-e", "crates/grpc_client/python/", "-e", "grpc_servicer/"],
        ["install", "--ignore-installed", "pytest"],
    ), args
elif args == ["-m", "venv", "--system-site-packages", ".venv"]:
    folder = Path(".venv/bin")
    folder.mkdir(parents=True)
    (folder / "activate").write_text("# Isolated command fixture; no environment install.\n")
elif args == ["-V"]:
    print("Python (controlled fixture)")
elif len(args) == 2 and args[0] == "-c":
    assert args[1] == (
        "import torch, vllm, smg_grpc_servicer; assert torch.version.hip; "
        "print('vllm', vllm.__version__)"
    ), args
elif args == ["-"]:
    # Execute the real helper's stdin canary, replacing only external imports.
    source = sys.stdin.read()
    original_import = builtins.__import__

    def fixture_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("torch", "mooncake.engine"):
            record({"canary_import": name})
            if name == "mooncake.engine":
                if os.environ["MISSING_MOONCAKE"] == "1":
                    raise ImportError("fixture Mooncake import unavailable")
                return types.SimpleNamespace(TransferEngine=object)
            return types.ModuleType("torch")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = fixture_import
    exec(compile(source, "<actual-vllm-canary-stdin>", "exec"), {"__name__": "__main__"})
else:
    raise AssertionError("Unexpected tool command: " + repr((role, args)))
"""


@pytest.mark.parametrize(
    "field,value,requested,unavailable",
    [
        pytest.param(None, "", False, True, id="default-no-mooncake"),
        pytest.param("E2E_VLLM_KV_BACKEND", "nixl", False, True, id="other-backend"),
        pytest.param("E2E_VLLM_KV_BACKEND", "mooncake", True, False, id="primary-present"),
        pytest.param("E2E_VLLM_KV_BACKEND", "mooncake", True, True, id="primary-missing"),
        pytest.param("E2E_KV_BACKEND", " MoOnCaKe ", True, False, id="generic-present"),
        pytest.param("E2E_KV_BACKEND", " MoOnCaKe ", True, True, id="generic-missing"),
        pytest.param(
            "E2E_VLLM_EXTRA_KV_BACKENDS", "nixl, mooncake ", True, False, id="extra-present"
        ),
        pytest.param(
            "E2E_VLLM_EXTRA_KV_BACKENDS", "nixl, mooncake ", True, True, id="extra-missing"
        ),
    ],
)
def test_vllm_requested_mooncake_canary_is_scoped_and_fail_closed(
    tmp_path, field, value, requested, unavailable
):
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("python3", "pip"):
        tool = binary / name
        tool.write_text(f"#!{sys.executable}\n" + VLLM_TOOL_FIXTURE)
        tool.chmod(0o755)
    cat = binary / "cat"
    cat.write_text(
        '#!/bin/sh\n[ "$#" = 1 ] && [ "$1" = /opt/rocm/.info/version ] || exit 55\n'
        'printf "fixture-ROCm\\n"\n'
    )
    cat.chmod(0o755)
    log = tmp_path / "tools.jsonl"
    handoff = tmp_path / "github.path"
    env = {
        "PATH": str(binary) + os.pathsep + os.defpath,
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_TOOL_LOG": str(log),
        "MISSING_MOONCAKE": "1" if unavailable else "0",
        "GITHUB_PATH": str(handoff),
    }
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    if field is not None:
        env[field] = value
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci_setup_vllm_rocm_image.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    events = [json.loads(line) for line in log.read_text().splitlines()]
    imports = [event["canary_import"] for event in events if "canary_import" in event]
    assert imports == (["torch", "mooncake.engine"] if requested else [])
    assert len([event for event in events if event.get("argv") == ["-"]]) == 1
    assert len([event for event in events if event.get("tool") == "pip"]) == 2
    if requested and unavailable:
        assert result.returncode != 0
        assert "fixture Mooncake import unavailable" in result.stderr
        assert not handoff.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert handoff.read_text() == str(tmp_path / ".venv/bin") + "\n"


def test_verify_only_dispatches_before_torch_install_preflight(tmp_path):
    report = tmp_path / "verify.json"
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=("--verify-only", str(report)),
        extra_env={"SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE, "FAIL_TORCH_PREFLIGHT": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text())["verified"] is True
    assert calls == [["-m", "pip", "check"]]
    assert not (tmp_path / "github.env").exists()


def test_native_repair_plan_selects_only_corrupt_same_version_tilelang(tmp_path, monkeypatch):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    assert verifier.native_repair_packages() == []
    (root / "tilelang/lib/libtilelang.so").write_bytes(b"corrupted same-version library")
    assert verifier.native_repair_packages() == ["tilelang"]
    assert native == [] and pip == []


def test_same_version_payload_is_replaced_without_touching_other_cohorts(tmp_path):
    payload = tmp_path / "tilelang-payload"
    payload.write_text("bad same-version payload")
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "NATIVE_CORRUPTION_FILE": str(payload),
        },
    )
    assert result.returncode == 0, result.stderr
    native = [call for call in calls if "--force-reinstall" in call]
    assert len(native) == 1 and native[0][-1:] == ["tilelang==0.1.15"]
    assert "--no-deps" in native[0]
    assert all(
        "apache-tvm-ffi==0.1.12" not in call
        and EXTENSION_SOURCE not in call
        and "torch" not in call
        for call in native
    )
    assert payload.read_text() == "repaired"


def test_healthy_native_cohort_is_not_reinstalled(tmp_path):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "NATIVE_REPAIR_PACKAGES": "[]",
        },
    )
    assert result.returncode == 0, result.stderr
    assert not any(
        "--force-reinstall" in call
        or "apache-tvm-ffi==0.1.12" in call
        or "tilelang==0.1.15" in call
        or EXTENSION_SOURCE in call
        for call in calls
    )


def test_record_validation_is_independent_of_current_pycache_prefix(tmp_path, monkeypatch):
    verifier, _, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "external-cache"))
    report = {}
    verifier.verify(report)
    assert report["pip_check"]["returncode"] == 0 and native and pip


def test_record_cache_without_allowed_source_is_still_rejected(tmp_path, monkeypatch):
    verifier, root, _, _ = _fixed_cohort_before_native(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "external-cache"))
    cached = root / "torch_c_dlpack_ext/__pycache__/orphan.cpython-313.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"fixture cache never executed")
    record = root / "torch_c_dlpack_ext-0.1.5.dist-info/RECORD"
    with record.open("a") as stream:
        stream.write(str(cached.relative_to(root)) + ",,\n")
    with pytest.raises(RuntimeError, match="Unhashed RECORD file"):
        verifier.verify({})


def _rewrite_fixture_record(root, folder):
    """Rehash only the test-authored payload, never installed runtime files."""
    record = folder / "RECORD"
    rows = []
    for name, _, _ in __import__("csv").reader(record.read_text().splitlines()):
        path = root / name
        if name == folder.name + "/RECORD":
            continue
        if not path.exists():
            continue
        body = path.read_bytes()
        encoded = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
        rows.append(f"{name},sha256={encoded},{len(body)}\n")
    record.write_text("".join(rows) + folder.name + "/RECORD,,\n")


@pytest.mark.parametrize(
    "attack",
    ["escaping-python", "foreign-torch", "symlink", "wrong-location", "wrong-version-escape"],
)
def test_unsafe_uninstall_admission_never_becomes_a_repair_plan(tmp_path, monkeypatch, attack):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    sentinel = root.parents[2] / "bin/python3"
    sentinel.write_bytes(b"owned fixture interpreter sentinel; never execute or uninstall")
    original = sentinel.read_bytes()
    folder = root / "tilelang-0.1.15.dist-info"
    if attack == "wrong-location":
        dist = next(d for d in verifier.md.distributions() if d.metadata["Name"] == "tilelang")
        dist._path = tmp_path / "foreign.dist-info"
    else:
        if attack == "wrong-version-escape":
            changed = root / "tilelang-0.1.14.dist-info"
            folder.rename(changed)
            dist = next(d for d in verifier.md.distributions() if d.metadata["Name"] == "tilelang")
            dist._path = changed
            old_version = verifier.md.version
            verifier.md.version = lambda name: "0.1.14" if name == "tilelang" else old_version(name)
            record = changed / "RECORD"
            record.write_text(record.read_text().replace(folder.name, changed.name))
            folder = changed
        if attack == "foreign-torch":
            target = root / "torch/__init__.py"
            target.parent.mkdir()
            target.write_bytes(b"owned fake Torch sentinel")
            name = "torch/__init__.py"
        elif attack == "symlink":
            outside = tmp_path / "outside"
            outside.mkdir()
            (outside / "payload").write_bytes(b"owned outside sentinel")
            (root / "tilelang/link").symlink_to(outside, target_is_directory=True)
            name = "tilelang/link/payload"
        else:
            name = "../../../bin/python3"
        with (folder / "RECORD").open("a") as stream:
            stream.write(name + ",,\n")
    with pytest.raises(RuntimeError, match="Unsafe"):
        verifier.native_repair_packages()
    assert native == [] and pip == []
    assert sentinel.read_bytes() == original


@pytest.mark.parametrize(
    "shape",
    ["binary-wheel", "wrong-direct-url", "impure-wheel", "missing-requested", "unrecorded-source"],
)
def test_dlpack_repair_and_final_verifier_share_the_complete_payload_contract(
    tmp_path, monkeypatch, shape
):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    folder = root / "torch_c_dlpack_ext-0.1.5.dist-info"
    if shape == "binary-wheel":
        addon = root / "torch_c_dlpack_ext/libtorch_c_dlpack_addon_torch2x.so"
        addon.write_bytes(b"\x7fELF owned fake addon; never load")
        with (folder / "RECORD").open("a") as stream:
            stream.write(str(addon.relative_to(root)) + ",,\n")
        (folder / "direct_url.json").unlink()
        (folder / "REQUESTED").unlink()
    elif shape == "wrong-direct-url":
        (folder / "direct_url.json").write_text(
            json.dumps({"url": "https://example.invalid/not-the-fixed-sdist"})
        )
    elif shape == "impure-wheel":
        (folder / "WHEEL").write_text(
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp313-cp313-manylinux_x86_64\n"
        )
    elif shape == "unrecorded-source":
        (root / "torch_c_dlpack_ext/unrecorded.py").write_text("# never import this fixture\n")
    else:
        (folder / "REQUESTED").unlink()
    _rewrite_fixture_record(root, folder)
    assert verifier.native_repair_packages() == ["torch-c-dlpack-ext"]
    with pytest.raises(RuntimeError):
        verifier.verify({})
    assert native == [] and pip == []


def test_setup_stops_before_any_pip_when_repair_admission_fails(tmp_path):
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        extra_env={
            "SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE,
            "IMAGE_REFLECTION_VERSION": "1.84.0",
            "REPAIR_PLAN_ERROR": "1",
        },
    )
    assert result.returncode != 0
    assert "Unsafe uninstall RECORD" in result.stderr
    assert calls == []
    assert not (tmp_path / "github.env").exists()


def _add_script_metadata_fixture(root, folder, kind, name):
    if kind == "dist-info-scripts":
        payload = folder / "scripts" / name
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"# owned metadata fixture, never execute\n")
    else:
        payload = folder / "entry_points.txt"
        if kind == "malformed-console":
            payload.write_text("[console_scripts]\npython3 without equals\n")
        else:
            payload.write_text(f"[{kind}]\n{name} = never_import_this_fixture:main\n")
    with (folder / "RECORD").open("a") as stream:
        stream.write(str(payload.relative_to(root)) + ",,\n")
    _rewrite_fixture_record(root, folder)


@pytest.mark.parametrize(
    "kind,name",
    [
        ("console_scripts", "python3"),
        ("gui_scripts", "python3.13"),
        ("console_scripts", "foreign-cli"),
        ("console_scripts", "../python3"),
        ("gui_scripts", "subdir/wrapper"),
        ("console_scripts", "/owned-fixture-only/foreign"),
        ("dist-info-scripts", "python3"),
        ("dist-info-scripts", "foreign-cli"),
        ("malformed-console", "python3"),
    ],
)
def test_repair_metadata_never_authorizes_foreign_script_deletion(
    tmp_path, monkeypatch, capsys, kind, name
):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    sentinels = [
        root.parents[2] / "bin" / leaf for leaf in ("python3", "python3.13", "foreign-cli")
    ]
    for sentinel in sentinels:
        sentinel.write_bytes(b"owned fake interpreter/foreign wrapper; never uninstall")
    before = {p: p.read_bytes() for p in sentinels}
    # A prior safe repair candidate must not leak a partial stdout plan if a
    # later candidate has unsafe metadata-derived pip deletion targets.
    (root / "tvm_ffi/core.abi3.so").write_bytes(b"earlier repairable owned fixture")
    (root / "tilelang/lib/libtilelang.so").write_bytes(b"later repairable owned fixture")
    _add_script_metadata_fixture(root, root / "tilelang-0.1.15.dist-info", kind, name)
    monkeypatch.setattr(sys, "argv", ["ci_verify_sglang_rocm_cohort.py", "--repair-plan"])
    with pytest.raises(RuntimeError, match="Automatic native repair refused"):
        verifier.main()
    assert capsys.readouterr().out == ""
    assert {p: p.read_bytes() for p in before} == before
    assert native == [] and pip == []


def test_wrong_version_still_requires_script_metadata_admission(tmp_path, monkeypatch, capsys):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    old = root / "tilelang-0.1.15.dist-info"
    folder = root / "tilelang-0.1.14.dist-info"
    old.rename(folder)
    dist = next(d for d in verifier.md.distributions() if d.metadata["Name"] == "tilelang")
    dist._path = folder
    version = verifier.md.version
    verifier.md.version = lambda name: "0.1.14" if name == "tilelang" else version(name)
    record = folder / "RECORD"
    record.write_text(record.read_text().replace(old.name, folder.name))
    _add_script_metadata_fixture(root, folder, "gui_scripts", "python3")
    monkeypatch.setattr(sys, "argv", ["ci_verify_sglang_rocm_cohort.py", "--repair-plan"])
    with pytest.raises(RuntimeError, match="Automatic native repair refused"):
        verifier.main()
    assert capsys.readouterr().out == ""
    assert native == [] and pip == []


def test_healthy_cohort_keeps_tvm_wrappers_and_non_script_plugin_metadata(tmp_path, monkeypatch):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    folder = root / "apache_tvm_ffi-0.1.12.dist-info"
    _add_script_metadata_fixture(root, folder, "console_scripts", "tvm-ffi-config")
    entrypoints = folder / "entry_points.txt"
    with entrypoints.open("a") as stream:
        stream.write("tvm-ffi-stubgen = never_import_this_fixture:main\n")
    _rewrite_fixture_record(root, folder)
    _add_script_metadata_fixture(
        root, root / "tilelang-0.1.15.dist-info", "fixture_plugins", "allowed-non-script-plugin"
    )
    assert verifier.native_repair_packages() == []
    assert native == [] and pip == []
    report = {}
    verifier.verify(report)
    assert report["pip_check"]["returncode"] == 0
    assert "never_import_this_fixture" not in native


def test_unqualified_tvm_script_metadata_holds_only_when_repair_is_needed(tmp_path, monkeypatch):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    _add_script_metadata_fixture(
        root, root / "apache_tvm_ffi-0.1.12.dist-info", "console_scripts", "tvm-ffi-config"
    )
    assert verifier.native_repair_packages() == []
    (root / "tvm_ffi/core.abi3.so").write_bytes(
        b"repair requires explicit script identity qualification"
    )
    with pytest.raises(RuntimeError, match="Automatic native repair refused"):
        verifier.native_repair_packages()
    assert native == [] and pip == []


@pytest.mark.parametrize(
    "name",
    [
        "__pycache__/six.cpython-313.pyc",
        "__pycache__/build_backend.py",
        "../__pycache__/build_backend.cpython-313.pyc",
    ],
)
def test_backend_cache_permission_does_not_allow_foreign_or_escaping_paths(
    tmp_path, monkeypatch, name
):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    folder = root / "torch_c_dlpack_ext-0.1.5.dist-info"
    with (folder / "RECORD").open("a") as stream:
        stream.write(name + ",,\n")
    with pytest.raises(RuntimeError, match="Unsafe"):
        verifier.native_repair_packages()
    assert native == [] and pip == []


@pytest.mark.parametrize("declared", ["different-package", None, ""])
def test_native_directory_identity_cannot_be_dropped_by_metadata_name(
    tmp_path, monkeypatch, declared
):
    import shutil

    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    dist = next(d for d in verifier.md.distributions() if d.metadata["Name"] == "tilelang")
    dist.metadata["Name"] = declared
    shutil.rmtree(root / "tilelang")
    with (dist._path / "RECORD").open("a") as stream:
        stream.write("../../../bin/python3,,\n")
    with pytest.raises(RuntimeError, match="Unsafe native distribution identity"):
        verifier.native_repair_packages()
    assert native == [] and pip == []


def test_missing_name_of_an_unrelated_distribution_does_not_reinstall_native_packages(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    actual = list(verifier.md.distributions())
    actual.append(SimpleNamespace(metadata={}, _path=root / "unrelated-1.dist-info"))
    verifier.md.distributions = lambda: actual
    assert verifier.native_repair_packages() == []
    assert native == [] and pip == []


@pytest.mark.parametrize("shape", ["null", "list", "archive-null", "hashes-null", "hashes-list"])
def test_malformed_archive_metadata_is_repairable_only_after_safety_admission(
    tmp_path, monkeypatch, shape
):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    data = {"url": verifier.SDIST_URL, "archive_info": {"hashes": {"sha256": verifier.SDIST_SHA}}}
    if shape == "null":
        data = None
    elif shape == "list":
        data = []
    elif shape == "archive-null":
        data["archive_info"] = None
    elif shape == "hashes-null":
        data["archive_info"]["hashes"] = None
    else:
        data["archive_info"]["hashes"] = []
    folder = root / "torch_c_dlpack_ext-0.1.5.dist-info"
    (folder / "direct_url.json").write_text(json.dumps(data))
    _rewrite_fixture_record(root, folder)
    assert verifier.native_repair_packages() == ["torch-c-dlpack-ext"]
    with pytest.raises(RuntimeError, match="Malformed"):
        verifier.verify({})
    assert native == [] and pip == []


@pytest.mark.parametrize(
    "kind,value",
    [
        ("dir_info", {}),
        ("vcs_info", {}),
        ("dir_info", {"editable": False}),
        ("vcs_info", {"vcs": "git"}),
    ],
)
def test_archive_origin_must_not_declare_another_origin_kind(tmp_path, monkeypatch, kind, value):
    verifier, root, native, pip = _fixed_cohort_before_native(tmp_path, monkeypatch)
    folder = root / "torch_c_dlpack_ext-0.1.5.dist-info"
    path = folder / "direct_url.json"
    data = json.loads(path.read_text())
    data[kind] = value
    path.write_text(json.dumps(data))
    _rewrite_fixture_record(root, folder)
    assert verifier.native_repair_packages() == ["torch-c-dlpack-ext"]
    with pytest.raises(RuntimeError, match="official sdist"):
        verifier.verify({})
    assert native == [] and pip == []


def test_verify_only_option_shaped_report_is_a_report_not_a_plan(tmp_path, monkeypatch):
    _pin_source_fixture(tmp_path, monkeypatch, 'IMAGE = "' + FIXED_IMAGE + '"')
    result, calls = run_installer(
        tmp_path,
        {},
        script="ci_setup_sglang_rocm_image.sh",
        arguments=("--verify-only", "--repair-plan"),
        extra_env={"SMG_SGLANG_ROCM_IMAGE": FIXED_IMAGE},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "--repair-plan").read_text())["verified"] is True
    assert calls == [["-m", "pip", "check"]]
    assert not (tmp_path / "github.env").exists()


def test_real_pip_backend_cache_and_directory_identity_are_safe(tmp_path, monkeypatch):
    """Real local pip/wheel artifacts, with no SDK, index, uninstall or target imports."""
    import csv
    import importlib.metadata
    import zipfile

    evidence = tmp_path / "real-pip-artifacts"
    evidence.mkdir()
    venv = tmp_path / "fresh-pip-venv"
    home = tmp_path / "private-home"
    home.mkdir()
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_NO_INPUT": "1",
    }
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    commands = []

    def run(label, argv):
        result = subprocess.run(
            argv, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=45
        )
        (evidence / (label + ".stdout")).write_text(result.stdout)
        (evidence / (label + ".stderr")).write_text(result.stderr)
        commands.append({"label": label, "argv": argv, "exit": result.returncode})
        (evidence / "commands.json").write_text(json.dumps(commands, indent=2))
        assert result.returncode == 0, result.stderr
        return result.stdout

    run("create-venv", [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", str(venv)])
    site = (
        venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    # Reuse immutable pip tooling, not another environment's SDKs. -B prevents
    # implicit bytecode writes there; pip's explicit default --compile remains on.
    pip_dist = importlib.metadata.distribution("pip")
    pip_source = Path(importlib.util.find_spec("pip").origin).resolve().parent
    (site / "pip").symlink_to(pip_source, target_is_directory=True)
    (site / Path(pip_dist._path).name).symlink_to(
        Path(pip_dist._path).resolve(), target_is_directory=True
    )
    py = str(venv / "bin/python")
    pip_version = run("pip-version", [py, "-I", "-B", "-m", "pip", "--version"])
    distname = "smg_cohort_cache_fixture-1.0.dist-info"
    wheel = evidence / "smg_cohort_cache_fixture-1.0-py3-none-any.whl"
    files = {
        "build_backend.py": b"VALUE = 'inert local wheel fixture; never import'\n",
        distname
        + "/METADATA": b"Metadata-Version: 2.1\nName: smg-cohort-cache-fixture\nVersion: 1.0\n",
        distname
        + "/WHEEL": b"Wheel-Version: 1.0\nGenerator: smg-owned-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    records = []
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, body in files.items():
            archive.writestr(zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0)), body)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
            records.append(f"{name},sha256={encoded},{len(body)}\n")
        archive.writestr(distname + "/RECORD", "".join(records) + distname + "/RECORD,,\n")
    run(
        "install-only-benign-local-wheel",
        [
            py,
            "-I",
            "-B",
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "install",
            "--no-index",
            "--no-deps",
            "--no-cache-dir",
            "--compile",
            str(wheel),
        ],
    )
    folder = site / distname
    record = (folder / "RECORD").read_text()
    (evidence / "PIP-GENERATED-RECORD.txt").write_text(record)
    cache_name = f"__pycache__/build_backend.{sys.implementation.cache_tag}.pyc"
    cache_rows = [row for row in csv.reader(record.splitlines()) if row[0] == cache_name]
    assert cache_rows == [[cache_name, "", ""]], record
    assert (site / cache_name).is_file()
    (evidence / "PIP-GENERATED-BACKEND.pyc").write_bytes((site / cache_name).read_bytes())
    verifier = load_cohort_verifier()
    verifier.verify_record(site, folder)
    verifier.validate_uninstall_record(site, folder, "torch_c_dlpack_ext")
    sentinel = venv / "bin/owned-foreign-sentinel"
    sentinel.write_bytes(b"do not remove; no uninstall will run")
    with (folder / "RECORD").open("a") as stream:
        stream.write("../../../bin/owned-foreign-sentinel,,\n")
    with pytest.raises(RuntimeError, match="Unsafe"):
        verifier.validate_uninstall_record(site, folder, "torch_c_dlpack_ext")
    assert sentinel.read_bytes() == b"do not remove; no uninstall will run"
    (folder / "RECORD").write_text(record)
    alias = site / "tilelang-0.1.15.dist-info"
    alias.mkdir()
    (alias / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: different-package\nVersion: 0.1.15\n"
    )
    (alias / "RECORD").write_text("../../../bin/owned-foreign-sentinel,,\n")
    # Actual pip uses the directory identity even when METADATA Name differs.
    from pip._internal.metadata import get_environment

    selected = get_environment([str(site)]).get_distribution("tilelang")
    assert selected is not None and selected.canonical_name == "tilelang"
    assert selected.raw_name == "different-package"
    distributions = list(importlib.metadata.distributions(path=[str(site)]))
    with pytest.raises(RuntimeError, match="Unsafe native distribution identity"):
        verifier.find_native_distributions(distributions, "tilelang")
    assert sentinel.read_bytes() == b"do not remove; no uninstall will run"
    (evidence / "receipt.json").write_text(
        json.dumps(
            {
                "pip_version": pip_version.strip(),
                "python": sys.version,
                "cache_tag": sys.implementation.cache_tag,
                "wheel_SHA256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "actual_root_backend_cache": cache_rows[0],
                "actual_pip_selected_canonical_name": selected.canonical_name,
                "actual_pip_selected_raw_name": selected.raw_name,
                "Source_policy_rejected_alias": True,
                "dangerous_RECORD_rejected_sentinel_unchanged": True,
                "real_uninstall_calls": 0,
                "testwheel_not_Torch_TVM_or_official_SDK": True,
            },
            indent=2,
        )
    )
