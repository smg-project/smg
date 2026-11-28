#!/usr/bin/env python3
"""Responses' common native phases; model installers and activation are separate.

The caller supplies an already owned A5 cohort and retains every model-specific
preflight, full Responses selection, strict process cleanup and nonce cleanup.
This component never discovers providers, patches an SDK or changes a gate.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCES = (
    "ci_sglang_responses_native_phase.py",
    "ci_sglang_cohort.py",
    "ci_sglang_native_paths.py",
    "ci_sglang_private_pip.py",
    "ci_verify_sglang_rocm_cohort.py",
    "ci_sglang_mxfp4_fallback.py",
)
MODELS = ("Qwen/Qwen2.5-14B-Instruct", "openai/gpt-oss-20b")
IMAGE = (
    "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
    "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def job_identity():
    result = {
        key: os.environ.get(key, "")
        for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "E2E_JOB_MARKER")
    }
    if any(not value or any(c in value for c in "\r\n") for value in result.values()):
        raise ValueError("Complete Responses job identity required")
    return result


def active_manifest(path, expected_sha, nonce):
    """Reject missing, foreign or changed cohorts before pip/native execution."""
    path = Path(path).absolute()
    body = path.read_bytes()
    if len(body) > 1024 * 1024 or hashlib.sha256(body).hexdigest() != expected_sha:
        raise ValueError("Responses cohort bytes differ")
    spec = json.loads(body)
    prefix = Path(spec["interpreter_prefix"])
    boot = sys.modules.get("ci_sglang_cohort")
    if (
        not (sys.flags.isolated and sys.flags.dont_write_bytecode)
        or path != path.resolve()
        or path.is_symlink()
        or prefix != prefix.resolve()
        or not path.is_relative_to(prefix)
        or prefix.is_relative_to(Path("/opt/venv"))
        or not spec.get("isolated_children")
        or spec.get("image") != IMAGE
        or os.environ.get("SMG_SGLANG_ROCM_IMAGE") != IMAGE
        or boot is None
        or sys.modules.get("sitecustomize") is not boot
        or Path(boot.__file__).with_name("smg-cohort.json") != path
        or boot.SMG_MANIFEST_SHA256 != expected_sha
        or Path(sys.prefix).resolve() != prefix
    ):
        raise ValueError("Matching active isolated Responses cohort required")
    binding = spec.get("responses_binding", {})
    if (
        binding.get("source_sha256") != source_hashes()
        or binding.get("workflow_sha256")
        != digest(Path(__file__).parents[1] / ".github/workflows/e2e-gpu-job.yml")
        or binding.get("job") != job_identity()
        or not isinstance(nonce, str)
        or len(nonce) != 32
        or any(c not in "0123456789abcdef" for c in nonce)
        or binding.get("nonce") != nonce
    ):
        raise ValueError("Responses Source, workflow or job nonce differs")
    if (
        prefix.parent.parent != Path(os.environ["RUNNER_TEMP"]).resolve()
        or str(Path(__file__).parent.resolve()) not in spec["import_roots"]
        or os.environ.get("E2E_ENGINE") != "sglang"
        or os.environ.get("E2E_GPU_VENDOR") != "amd"
    ):
        raise ValueError("Explicit owned AMD SGLang Responses context required")
    # No model flag is changed here. A5's strict observer retains its original
    # PRE/POST contract; only the separately reviewed model owner may install it.
    return {"path": path, "sha256": expected_sha, "spec": spec}


def readonly_borrowed_providers(spec):
    caps = int(
        next(
            line.split()[1]
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("CapEff:")
        ),
        16,
    )
    if caps & (1 << 21):
        raise ValueError("Provider mounts must not be remountable")
    roots = {Path("/opt/venv"), *(Path(p) for p in spec["borrowed_sites"])}
    scope = Path(spec["interpreter_prefix"]).parent
    roots.update(
        Path(item["package_root"])
        for item in spec["packages"]
        if not Path(item["package_root"]).is_relative_to(scope)
    )
    for root in roots:
        if (
            root != root.resolve()
            or not root.is_dir()
            or not os.statvfs(root).f_flag & os.ST_RDONLY
        ):
            raise ValueError("Actual read-only borrowed provider required")


def install_native(cohort, directory):
    """Use A5's owned pip destination for the three original native inputs."""
    import ci_sglang_private_pip as private_pip
    import ci_verify_sglang_rocm_cohort as native

    spec = cohort["spec"]
    if spec.get("image") != native.IMAGE or os.environ.get("SMG_SGLANG_ROCM_IMAGE") != native.IMAGE:
        raise ValueError("Original fixed-image cohort required")
    readonly_borrowed_providers(spec)
    env_file = Path(os.environ["GITHUB_ENV"])
    if env_file.is_symlink() or env_file.absolute() != env_file.resolve() or not env_file.is_file():
        raise ValueError("Explicit job environment file required")
    arguments = [
        "--no-cache-dir",
        "--no-build-isolation",
        "--no-binary=torch-c-dlpack-ext",
        "apache-tvm-ffi==0.1.12",
        "tilelang==0.1.15",
        "torch-c-dlpack-ext @ " + native.SDIST_URL + "#sha256=" + native.SDIST_SHA,
    ]
    command, environment = private_pip.install_command(cohort["path"], arguments)
    subprocess.run(
        [
            command[0],
            "-c",
            "import torch; a=getattr(torch.Tensor,'__dlpack_c_exchange_api__',None); "
            "assert torch.version.hip and a is not None and type(a).__name__=='PyCapsule'",
        ],
        env=environment,
        check=True,
        timeout=60,
    )
    with (directory / "responses-manifest-before-install.json").open("x") as saved:
        saved.write(cohort["path"].read_text())
    for stage, complete in enumerate(
        (["--force-reinstall", "--no-deps", *arguments], arguments), 1
    ):
        private_pip.install(cohort["path"], complete, cwd=Path(__file__).parents[1])
        # Persist each validated commit before a later resolver can fail. A
        # failure never invents a digest from arbitrary current manifest bytes.
        previous = cohort["sha256"]
        rebind_native(cohort, native)
        receipt = {
            "manifest": str(cohort["path"]),
            "previous_sha256": previous,
            "sha256": cohort["sha256"],
            "responses_binding": cohort["spec"]["responses_binding"],
            "consumer_sha256": cohort["spec"].get("responses_consumer_sha256"),
            "producer_contract_sha256": cohort["spec"].get("responses_producer_contract_sha256"),
        }
        with (directory / f"responses-native-manifest-commit-{stage}.json").open("x") as output:
            output.write(json.dumps(receipt, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        with env_file.open("a") as output:
            for key in (
                "SMG_RESPONSES_SHA256",
                "SMG_RESPONSES_COHORT_SHA256",
                "SMG_GPTOSS_COHORT_MANIFEST_SHA256",
            ):
                output.write(key + "=" + cohort["sha256"] + "\n")
            output.flush()
            os.fsync(output.fileno())


def rebind_native(cohort, native):
    """Validate installed data without importing it, retaining every other pin."""
    if digest(cohort["path"]) != cohort["sha256"]:
        raise ValueError("Responses manifest changed during native installation")
    spec = cohort["spec"]
    root = (
        Path(spec["install_prefix"])
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    if root != root.resolve() or not root.is_dir():
        raise ValueError("Unaliased owned native data root required")
    packages, record_names = [], set()
    for distribution, version, module in (
        ("apache_tvm_ffi", "0.1.12", "tvm_ffi"),
        ("tilelang", "0.1.15", "tilelang"),
        ("torch_c_dlpack_ext", "0.1.5", "torch_c_dlpack_ext"),
    ):
        source = root / module / "__init__.py"
        metadata = root / f"{distribution}-{version}.dist-info/METADATA"
        if any(p != p.resolve() or not p.is_file() for p in (source, metadata)):
            raise ValueError("Unaliased native source and metadata required")
        if digest(metadata) != native.ANCHORS[str(metadata.relative_to(root))]:
            raise ValueError("Native metadata anchor differs")
        consoles = (
            ("../../../bin/tvm-ffi-config", "../../../bin/tvm-ffi-stubgen")
            if module == "tvm_ffi"
            else ()
        )
        record_names.update(
            native.verify_record(
                root, metadata.parent, consoles, console_dir=root.parents[2] / "bin"
            )
        )
        packages.append(
            dict(
                module=module,
                distribution=distribution.replace("_", "-"),
                version=version,
                package_root=str(root),
                module_path=str(source),
                metadata_path=str(metadata),
                source_sha256=digest(source),
                metadata_sha256=digest(metadata),
            )
        )
    if not set(native.ANCHORS) <= record_names:
        raise ValueError("Official native anchor missing from installed RECORD")
    for name, expected in native.ANCHORS.items():
        path = root / name
        if path != path.resolve() or not path.is_file() or digest(path) != expected:
            raise ValueError("Official native payload anchor differs: " + name)
    names = {p["module"] for p in packages}
    updated = {
        **spec,
        "packages": [p for p in spec["packages"] if p["module"] not in names] + packages,
    }
    pending = cohort["path"].with_suffix(".next")
    with pending.open("x") as output:
        output.write(json.dumps(updated, indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(pending, cohort["path"])
    cohort.update(spec=updated, sha256=digest(cohort["path"]))


def verify_phase(phase, directory, cohort):
    import ci_verify_sglang_rocm_cohort as native

    report = {"verified": False}
    try:
        native.verify(report, private_cohort=True)
        report["verified"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        with (directory / ("responses-native-" + phase + ".json")).open("x") as output:
            output.write(json.dumps(report, indent=2) + "\n")
    contract = {
        "native": report,
        "manifest_sha256": cohort["sha256"],
        "source_sha256": source_hashes(),
        "job": job_identity(),
    }
    if phase == "final":
        prepared = json.loads((directory / "responses-cohort-prepared.json").read_text())
        if prepared.get("status") != "passed" or prepared.get("contract") != contract:
            raise ValueError("Prepared/final Responses native contract differs")
    return {"status": "passed", "phase": phase, "contract": contract}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("install", "prepared", "final"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args(argv)
    args.directory.mkdir(exist_ok=True)
    result = {
        "status": "failed",
        "phase": args.phase,
        "scope": "generic_native_component_only",
        "model_qualification_not_proved": list(MODELS),
    }
    with (args.directory / ("responses-cohort-" + args.phase + ".json")).open("x") as output:
        try:
            cohort = active_manifest(args.manifest, args.sha256, args.nonce)
            if args.phase == "install":
                install_native(cohort, args.directory)
                result["status"] = "passed"
            else:
                result.update(verify_phase(args.phase, args.directory, cohort))
        except Exception as error:
            result["error"] = {"type": type(error).__name__, "message": str(error)}
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
