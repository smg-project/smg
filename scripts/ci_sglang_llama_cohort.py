#!/usr/bin/env python3
"""Create the ordinary Llama cohort from explicit fixed-image provider data.

Run with -I -S -B. Provider discovery and SDK/native execution are not performed.
"""

import argparse
import hashlib
import json
import os
import secrets
import shutil
import sys
import types
from email.parser import Parser
from pathlib import Path
from urllib.parse import unquote, urlparse

SOURCES = (
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_file(path, expected):
    path = Path(path).absolute()
    if (
        len(expected) != 64
        or any(c not in "0123456789abcdef" for c in expected)
        or path != path.resolve()
        or not path.is_file()
        or path.is_symlink()
        or digest(path) != expected
    ):
        raise ValueError("Provider Source bytes/path differ: " + str(path))
    return path


def source_module(path, expected):
    """Execute the verified Source bytes without consulting an existing pyc."""
    path = Path(path).absolute()
    name = path.stem
    if name in sys.modules:
        raise ValueError("Creator helper was already loaded: " + name)
    if path != path.resolve() or not path.is_file() or path.is_symlink():
        raise ValueError("Unaliased creator helper Source required")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Creator helper Source bytes differ: " + name)
    code = compile(body, str(path), "exec", dont_inherit=True, optimize=0)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def regular_tree(root):
    root = Path(root)
    if root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise ValueError("Unaliased provider directory required")
    total = 0
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError("Linked or special provider payload")
            if path.is_file():
                total += path.stat().st_size
    if total > 512 * 1024 * 1024:
        raise ValueError("Provider tree exceeds the bounded staging size")


def copy_tree(source, target):
    shutil.copytree(
        source,
        target,
        copy_function=shutil.copy2,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    for parent, directories, files in os.walk(target):
        Path(parent).chmod(0o755)
        for name in files:
            path = Path(parent) / name
            path.chmod(0o644 | (path.stat().st_mode & 0o111))


def readonly_providers(roots):
    """A borrowed dependency view alone does not prevent parent-prefix writes."""
    status = Path("/proc/self/status").read_text()
    caps = int(
        next(line.split()[1] for line in status.splitlines() if line.startswith("CapEff:")), 16
    )
    if caps & (1 << 21):
        raise ValueError("Provider mounts must not be remountable by this job")
    for root in roots:
        if not os.statvfs(root).f_flag & os.ST_RDONLY:
            raise ValueError("Read-only provider prefix required before any installer")


def editable_projects(checkout, roots, canonical):
    """Bind the two actual -e projects to their declared setuptools import roots."""
    import tomllib

    declared = {}
    for relative, module in (
        ("crates/grpc_client/python", "smg_grpc_proto"),
        ("grpc_servicer", "smg_grpc_servicer"),
    ):
        project = canonical(checkout / relative)
        config = project / "pyproject.toml"
        if config.is_symlink() or config.resolve() != config.absolute():
            raise ValueError("Editable project declaration must be unaliased")
        data = tomllib.loads(config.read_text())
        find = data["tool"]["setuptools"]["packages"]["find"]
        if find.get("where") != ["."] or module + "*" not in find.get("include", []):
            raise ValueError("Editable package root differs from its reviewed declaration")
        source = project / module / "__init__.py"
        if source.is_symlink() or not source.is_file() or source.resolve() != source.absolute():
            raise ValueError("Editable package source must have its declared origin")
        if project not in roots:
            raise ValueError("Editable project is absent from explicit import roots")
        declared[str(project)] = [str(project)]
    return declared


def job_identity():
    identity = {
        name: os.environ.get(name, "")
        for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "E2E_JOB_MARKER")
    }
    if any(not value or any(c in value for c in "\r\n") for value in identity.values()):
        raise ValueError("Explicit owned CI job identity required")
    return identity


def cleanup(directory, nonce):
    """Remove only the exact nonce created by this job, after process reaping."""
    root = Path(directory).absolute()
    temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    if root != root.resolve() or root.parent != temp or not root.is_dir():
        raise ValueError("Unaliased immediate owned temporary cohort required")
    marker = root / "ordinary-owner.json"
    if marker.is_symlink() or not marker.is_file() or not nonce:
        raise ValueError("Known owner marker and nonce required; never adopt by name")
    owner = json.loads(marker.read_text())
    if owner != {"root": str(root), "nonce": nonce, "job": job_identity()}:
        raise ValueError("Owned cohort identity differs; preserve unknown directory")
    shutil.rmtree(root)
    return {"removed_owned_cohort": str(root), "exists": root.exists()}


def build(input_path, expected_sha, directory):
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise ValueError("Initial bootstrap requires -I -S -B before any provider startup")
    body = Path(input_path).read_bytes()
    if len(body) > 1024 * 1024 or hashlib.sha256(body).hexdigest() != expected_sha:
        raise ValueError("Provider input bytes differ")
    packet = json.loads(body)
    scripts = Path(__file__).resolve().parent
    hashes = {name: digest(scripts / name) for name in SOURCES}
    workflow_sha = digest(scripts.parent / ".github/workflows/e2e-gpu-job.yml")
    if packet.get("workflow_sha256") != workflow_sha:
        raise ValueError("Private workflow Source contract differs")
    if packet.get("source_sha256") != hashes:
        raise ValueError("Bootstrap/profile Source contract differs")
    if packet.get("schema") != "smg-ordinary-sg-provider-v1":
        raise ValueError("Explicit ordinary provider schema required")
    if (
        not isinstance(packet.get("creator_core_sha256"), str)
        or len(packet["creator_core_sha256"]) != 64
    ):
        raise ValueError("Complete creator ELF identity required")
    if not packet.get("base_sites") or set(packet.get("providers", {})) != {
        "torch",
        "sglang",
        "pip",
    }:
        raise ValueError("Complete explicit original provider set required")
    if not packet.get("backend_hook", {}).get("path") or not packet["backend_hook"].get("sha256"):
        raise ValueError("Actual backend Source identity required")

    # -B prevents writes, but still permits reading timestamp-valid old pyc.
    # Use the same verified bytes for these first imports, before the new
    # interpreter's empty execution-cache redirect exists.
    cohort = source_module(scripts / "ci_sglang_cohort.py", hashes["ci_sglang_cohort.py"])
    sdk = source_module(
        scripts / "ci_sglang_mxfp4_fallback.py", hashes["ci_sglang_mxfp4_fallback.py"]
    )
    native = source_module(
        scripts / "ci_verify_sglang_rocm_cohort.py", hashes["ci_verify_sglang_rocm_cohort.py"]
    )

    if (
        packet.get("image") != native.IMAGE
        or packet.get("python_version") != list(sys.version_info[:2])
        or digest(Path("/proc/self/exe").resolve()) != packet["creator_core_sha256"]
        or os.environ.get("E2E_ENGINE") != "sglang"
        or os.environ.get("E2E_GPU_VENDOR") != "amd"
        or os.environ.get(sdk.ROW_ENV, "0") != "0"
        or os.environ.get("SGLANG_GPTOSS_MXFP4_BF16_FALLBACK", "0") != "0"
        or bool(os.environ.get("SMG_GPTOSS_COHORT_MANIFEST", ""))
        or os.environ.get("E2E_SGLANG_GPTOSS_MXFP4_BF16", "0") != "0"
        or os.environ.get("SMG_SGLANG_ROCM_IMAGE") != native.IMAGE
        or str(native.CONSTRAINT) not in os.environ.get("PIP_CONSTRAINT", "").split()
    ):
        raise ValueError("Explicit fixed-image creator and original inputs required")
    bases = [cohort.canonical(path) for path in packet["base_sites"]]
    packages = []
    for module, version in (
        ("torch", "2.11.0+rocm10.0.0"),
        ("sglang", "0.5.21"),
        ("pip", "26.2.1"),
    ):
        item = dict(packet["providers"][module])
        source = exact_file(item["module_path"], item["source_sha256"])
        metadata = exact_file(item["metadata_path"], item["metadata_sha256"])
        info = Parser().parsestr(metadata.read_text())
        if info["Name"].lower() != module or info["Version"] != version:
            raise ValueError("Original provider metadata identity differs")
        if metadata.parent.parent not in bases:
            raise ValueError("Provider metadata is outside declared base sites")
        if module != "sglang" and source != metadata.parent.parent / module / "__init__.py":
            raise ValueError("Provider module/metadata origins differ")
        item.update(
            module=module,
            distribution=module,
            version=version,
            package_root=str(metadata.parent.parent),
        )
        packages.append(item)
    sg = next(item for item in packages if item["module"] == "sglang")
    sg_source = Path(sg["module_path"]).parent
    sg_metadata = Path(sg["metadata_path"]).parent
    regular_tree(sg_source)
    regular_tree(sg_metadata)
    direct = json.loads((sg_metadata / "direct_url.json").read_text())
    editable = urlparse(direct.get("url", ""))
    if (
        editable.scheme != "file"
        or Path(unquote(editable.path)).resolve() != sg_source.parent
        or direct.get("dir_info", {}).get("editable") is not True
    ):
        raise ValueError("Explicit SGLang editable origin required")
    if sdk.source_hashes(sg_source) != {name: pair[0] for name, pair in sdk.FILES.items()}:
        raise ValueError("Original fixed-image SDK source pair differs")
    readonly_providers({Path("/opt/venv"), sg_source, *bases})
    hook = packet["backend_hook"]
    exact_file(hook["path"], hook["sha256"])
    root = Path(directory).absolute()
    cohort.canonical(root.parent)
    if (
        root.exists()
        or root.is_symlink()
        or root.is_relative_to(Path("/opt/venv"))
        or any(root.is_relative_to(path) for path in [sg_source, sg_metadata, *bases])
        or any(c in str(root) for c in "\r\n")
    ):
        raise ValueError("Exclusive owned cohort directory required")
    identity = job_identity()
    if root.parent != Path(os.environ["RUNNER_TEMP"]).resolve():
        raise ValueError("Cohort must be immediately under owned RUNNER_TEMP")
    env_file = Path(os.environ["GITHUB_ENV"])
    if env_file.is_symlink() or env_file.absolute() != env_file.resolve() or not env_file.is_file():
        raise ValueError("Explicit unaliased job environment file required")
    root.mkdir(mode=0o700)
    nonce = secrets.token_hex(16)
    with (root / "ordinary-owner.json").open("x") as handle:
        json.dump({"root": str(root), "nonce": nonce, "job": identity}, handle)
    # Publish the exact owned handle before staging, so failures can clean up.
    with env_file.open("a") as handle:
        handle.write("SMG_SGLANG_LLAMA_SCOPE=" + str(root) + "\n")
        handle.write("SMG_SGLANG_LLAMA_SCOPE_NONCE=" + nonce + "\n")
    (root / "provider-input.json").write_bytes(body)
    sdk_parent = root / "sdk"
    sdk_parent.mkdir()
    copy_tree(sg_source, sdk_parent / "sglang")
    copy_tree(sg_metadata, sdk_parent / sg_metadata.name)
    sg.update(
        module_path=str(sdk_parent / "sglang/__init__.py"),
        metadata_path=str(sdk_parent / sg_metadata.name / "METADATA"),
        package_root=str(sdk_parent),
    )
    site = (
        root / f"native/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    site.mkdir(parents=True)
    (root / "native/bin").mkdir()
    for name in ("home", "tmp", "cache", "build-tracker"):
        (root / name).mkdir()
    roots = [site, sdk_parent, scripts]
    checkout = scripts.parent
    for relative in ("crates/grpc_client/python", "grpc_servicer", "e2e_test", "clients/python"):
        for path in (checkout / relative, checkout / relative / "src"):
            if path.is_dir():
                roots.append(cohort.canonical(path))
    python = cohort.create(
        root / "interpreter",
        roots,
        packages,
        borrowed_sites=bases,
        isolated_children=True,
        backend_hook=hook,
        editable_projects=editable_projects(checkout, roots, cohort.canonical),
    )
    manifest = next((root / "interpreter/lib").glob("python*/site-packages/smg-cohort.json"))
    spec = json.loads(manifest.read_text())
    spec.update(
        image=native.IMAGE,
        install_prefix=str(root / "native"),
        llama_source_sha256=hashes,
        workflow_sha256=workflow_sha,
        provider_input_sha256=expected_sha,
        provider_provenance=packet["providers"],
    )
    spec["job_marker"] = os.environ.get("E2E_JOB_MARKER", "")
    manifest.write_text(json.dumps(spec, indent=2) + "\n")
    # Bare pip console calls must use the same destination policy as -m callers.
    import shlex

    shim = Path(python).with_name("pip")
    shim.write_text(
        "#!/bin/bash -p\nexec "
        + shlex.quote(str(python))
        + " -m ci_sglang_private_pip --manifest "
        + shlex.quote(str(manifest))
        + ' -- "$@"\n'
    )
    shim.chmod(0o755)
    entry_code = (
        "import sys\nfrom importlib.metadata import entry_points\n"
        "name=sys.argv.pop(1)\nentries=list(entry_points(group='console_scripts',name=name))\n"
        "if len(entries)!=1: raise RuntimeError('Ambiguous or missing cohort console entry')\n"
        "sys.argv[0]=name\nraise SystemExit(entries[0].load()())\n"
    )
    consoles = {}
    for name in ("pytest", "hf"):
        console = Path(python).with_name(name)
        console.write_text(
            "#!/bin/bash -p\nexec "
            + shlex.quote(str(python))
            + " -c "
            + shlex.quote(entry_code)
            + " "
            + shlex.quote(name)
            + ' "$@"\n'
        )
        console.chmod(0o755)
        consoles[name] = digest(console)
    spec["console_shim_sha256"] = consoles
    spec["pip_shim_sha256"] = digest(shim)
    manifest.write_text(json.dumps(spec, indent=2) + "\n")
    return {
        "manifest": str(manifest),
        "sha256": digest(manifest),
        "python": str(python),
        "scope": "Provider bytes staged; no SDK/pip/backend/native execution or acceptance",
    }


def packet_image():
    return (
        "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
        "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cleanup", type=Path)
    parser.add_argument("--nonce")
    args = parser.parse_args(argv)
    if args.cleanup is not None:
        if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
            parser.error("Cleanup also requires -I -S -B")
        print(json.dumps(cleanup(args.cleanup, args.nonce)))
        return 0
    if args.input is None or not args.sha256 or args.directory is None:
        parser.error("Complete packet path/hash and exclusive directory required")
    result = build(args.input, args.sha256, args.directory)
    if args.output:
        with args.output.open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
    if os.environ.get("GITHUB_ENV"):
        with Path(os.environ["GITHUB_ENV"]).open("a") as handle:
            handle.write("SMG_SGLANG_LLAMA_COHORT_MANIFEST=" + result["manifest"] + "\n")
            handle.write("SMG_SGLANG_LLAMA_COHORT_MANIFEST_SHA256=" + result["sha256"] + "\n")
            handle.write("_PIP_USE_IMPORTLIB_METADATA=1\n")
            handle.write("E2E_SGLANG_GPTOSS_MXFP4_BF16=0\n")
            handle.write("SGLANG_GPTOSS_MXFP4_BF16_FALLBACK=0\n")
            handle.write("E2E_ENGINE=sglang\n")
            handle.write("E2E_GPU_VENDOR=amd\n")
            handle.write("SMG_SGLANG_ROCM_IMAGE=" + packet_image() + "\n")
    if os.environ.get("GITHUB_PATH"):
        with Path(os.environ["GITHUB_PATH"]).open("a") as handle:
            prefix = Path(result["python"]).parent.parent
            handle.write(str(prefix.parent / "native/bin") + "\n")
            handle.write(str(prefix / "bin") + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
