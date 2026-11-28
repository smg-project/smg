#!/usr/bin/env python3
"""Ordinary Llama private install/strict phase consumer; no20B producer calls."""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import ci_sglang_mxfp4_fallback as sdk
import ci_verify_sglang_rocm_cohort as native

SDIST = "torch-c-dlpack-ext @ " + native.SDIST_URL + "#sha256=" + native.SDIST_SHA
PRIVATE_SOURCES = (
    "ci_sglang_llama_cohort.py",
    "ci_sglang_llama_profile.py",
    "ci_sglang_mxfp4_fallback.py",
    "ci_verify_sglang_rocm_cohort.py",
    "ci_sglang_cohort.py",
    "ci_sglang_native_paths.py",
    "ci_sglang_private_pip.py",
    "ci_setup_sglang_rocm_image.sh",
    "ci_install_e2e_deps.sh",
)


def native_site(spec):
    version = spec.get("python_version")
    if (
        not isinstance(version, list)
        or len(version) != 2
        or any(type(value) is not int or value < 0 for value in version)
    ):
        raise ValueError("Explicit cohort Python version required")
    return Path(spec["install_prefix"]) / f"lib/python{version[0]}.{version[1]}/site-packages"


def available_scope(prefix):
    marker = Path(prefix).parent / ".private-pip-transaction"
    if marker.exists() or marker.is_symlink():
        raise ValueError("Failed or in-flight private install scope must be discarded")


def publish_manifest(path, spec):
    """Publish one complete successful native contract before the next child."""
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=".ordinary-manifest-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(spec, indent=2) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def private_manifest(path, expected_sha, *, active):
    body = path.read_bytes()
    if len(body) > 1024 * 1024 or hashlib.sha256(body).hexdigest() != expected_sha:
        raise ValueError("Private cohort manifest bytes differ")
    spec = json.loads(body)
    if spec.get("image") != native.IMAGE or not spec.get("isolated_children"):
        raise ValueError("Explicit fixed-image private cohort required")
    sources = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in PRIVATE_SOURCES
    }
    if spec.get("llama_source_sha256") != sources:
        raise ValueError("Private profile/helper Source differs")
    workflow_sha = hashlib.sha256(
        (Path(__file__).parents[1] / ".github/workflows/e2e-gpu-job.yml").read_bytes()
    ).hexdigest()
    if spec.get("workflow_sha256") != workflow_sha:
        raise ValueError("Private workflow Source differs")
    prefix = Path(spec["interpreter_prefix"]).resolve()
    native_site(spec)
    if active and spec["python_version"] != list(sys.version_info[:2]):
        raise ValueError("Active ordinary interpreter minor differs from its cohort")
    available_scope(prefix)
    if not path.resolve().is_relative_to(prefix):
        raise ValueError("Private manifest must belong to its interpreter")
    if str(Path(__file__).parent.resolve()) not in spec["import_roots"]:
        raise ValueError("Private -I interpreter lacks the exact profile Source root")
    providers = {p["module"]: p for p in spec["packages"]}
    if (
        providers.get("torch", {}).get("version") != "2.11.0+rocm10.0.0"
        or providers.get("sglang", {}).get("version") != "0.5.21"
    ):
        raise ValueError("Explicit original Torch/SGLang provider entries required")
    sglang = providers["sglang"]
    sdk_source = Path(sglang["module_path"]).resolve()
    if not sdk_source.is_relative_to(prefix.parent) or sdk_source.is_relative_to(prefix):
        raise ValueError("An owned staged SGLang SDK copy is required")
    if active:
        boot = sys.modules.get("ci_sglang_cohort")
        if (
            boot is None
            or sys.modules.get("sitecustomize") is not boot
            or Path(boot.__file__).with_name("smg-cohort.json").resolve() != path.resolve()
            or boot.SMG_MANIFEST_SHA256 != expected_sha
        ):
            raise ValueError("Matching active private interpreter required")
    return {
        "path": path,
        "sha256": expected_sha,
        "spec": spec,
        "sources": sources,
        "workflow_sha256": workflow_sha,
    }


def prepare_private(output, cohort):
    """Install the original native inputs into an already staged owned cohort.

    Provider discovery and copying the editable SDK are separate required
    inputs. Missing or foreign provider data never falls back to base mutation.
    """
    import ci_sglang_private_pip as private_pip

    context()
    arguments = [
        "--no-cache-dir",
        "--no-build-isolation",
        "--no-binary=torch-c-dlpack-ext",
        "apache-tvm-ffi==0.1.12",
        "tilelang==0.1.15",
        SDIST,
    ]
    checkout = Path(__file__).resolve().parents[1]
    available_scope(Path(cohort["spec"]["interpreter_prefix"]))
    command, environment = private_pip.install_command(cohort["path"], arguments, cwd=checkout)
    python = command[0]
    subprocess.run(
        [
            python,
            "-c",
            "import torch; api=getattr(torch.Tensor,'__dlpack_c_exchange_api__',None); "
            "assert torch.version.hip and api is not None and type(api).__name__=='PyCapsule', "
            "'ROCm Torch builtin DLPack C API required before native repair'",
        ],
        env=environment,
        check=True,
    )
    # Ordinary Llama keeps the exact original SDK PRE Source; no SDK patch producer.
    # --prefix alone may accept a same-version borrowed distribution without
    # copying it. Materialize only these original three inputs in our scheme,
    # then preserve their complete normal dependency resolution below.
    private_pip.install(
        cohort["path"],
        ["--force-reinstall", "--no-deps", *arguments],
        cwd=checkout,
    )
    private_pip.install(cohort["path"], arguments, cwd=checkout)
    spec = cohort["spec"]
    root = native_site(spec)
    packages = []
    for distribution, version, module in (
        ("apache_tvm_ffi", "0.1.12", "tvm_ffi"),
        ("tilelang", "0.1.15", "tilelang"),
        ("torch_c_dlpack_ext", "0.1.5", "torch_c_dlpack_ext"),
    ):
        source = root / module / "__init__.py"
        metadata = root / f"{distribution}-{version}.dist-info/METADATA"
        metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
        if metadata_sha != native.ANCHORS[str(metadata.relative_to(root))]:
            raise ValueError("Installed native metadata anchor differs")
        packages.append(
            {
                "module": module,
                "distribution": distribution.replace("_", "-"),
                "version": version,
                "package_root": str(root),
                "module_path": str(source),
                "metadata_path": str(metadata),
                "metadata_sha256": metadata_sha,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    names = {p["module"] for p in packages}
    spec["packages"] = [p for p in spec["packages"] if p["module"] not in names] + packages
    # Preserve the exact initial input before replacing the owned active spec.
    with output.with_name("llama-private-input-before-install.json").open("x") as saved:
        saved.write(cohort["path"].read_text())
    publish_manifest(cohort["path"], spec)
    updated = hashlib.sha256(cohort["path"].read_bytes()).hexdigest()
    # Later diagnostics may fail after this contract is already committed.
    # Export its exact identity before starting another child.
    with Path(os.environ["GITHUB_ENV"]).open("a") as handle:
        handle.write("SMG_SGLANG_LLAMA_COHORT_MANIFEST_SHA256=" + updated + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    version = subprocess.run(
        [
            python,
            "-c",
            "from importlib.metadata import version; print(version('opentelemetry-api'))",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not version or any(c not in "0123456789abcdefghijklmnopqrstuvwxyz.+-" for c in version):
        raise ValueError("Invalid image telemetry version")
    with Path(os.environ["GITHUB_ENV"]).open("a") as handle:
        handle.write("SMG_CI_OTEL_API_VERSION=" + version + "\n")


def context():
    if (
        os.environ.get("E2E_ENGINE") != "sglang"
        or os.environ.get("E2E_GPU_VENDOR") != "amd"
        or os.environ.get("SMG_SGLANG_ROCM_IMAGE") != native.IMAGE
        or os.environ.get("E2E_SGLANG_GPTOSS_MXFP4_BF16", "0") != "0"
        or os.environ.get(sdk.ROW_ENV, "0") != "0"
        or os.environ.get("SGLANG_GPTOSS_MXFP4_BF16_FALLBACK", "0") != "0"
        or bool(os.environ.get("SMG_GPTOSS_COHORT_MANIFEST", ""))
    ):
        raise ValueError("Explicit ordinary AMD SGLang context required,20B flags forbidden")
    if md.version("sglang") != "0.5.21" or md.version("torch") != "2.11.0+rocm10.0.0":
        raise ValueError("Unsupported original image cohort")


def verify(phase, directory, *, cohort):
    context()
    report = {"verified": False}
    try:
        native.verify(report, private_cohort=True)
        report["verified"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        with (directory / ("sglang-native-environment-" + phase + ".json")).open("x") as handle:
            handle.write(json.dumps(report, indent=2) + "\n")
    contract = {
        "native": report,
        "manifest_sha256": cohort["sha256"],
        "source_sha256": cohort["sources"],
        "workflow_sha256": cohort["workflow_sha256"],
    }
    if phase == "final":
        prepared = json.loads((directory / "llama-cohort-prepared.json").read_text())
        if prepared.get("status") != "passed" or prepared.get("contract") != contract:
            raise ValueError("Prepared/final ordinary cohort differs")
    return {"status": "passed", "phase": phase, "contract": contract}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("install", "prepared", "final"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    args.directory.mkdir(exist_ok=True)
    with (args.directory / ("llama-cohort-" + args.phase + ".json")).open("x") as handle:
        result = {"status": "failed", "phase": args.phase}
        try:
            context()
            cohort = private_manifest(args.manifest, args.sha256, active=True)
            if args.phase == "install":
                prepare_private(args.directory / "llama-install.json", cohort)
                result["status"] = "passed"
            else:
                result = verify(args.phase, args.directory, cohort=cohort)
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
