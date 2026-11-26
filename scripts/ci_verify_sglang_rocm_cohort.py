#!/usr/bin/env python3
"""Verify the fixed SGLang ROCm cohort without installing or patching packages."""

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import sysconfig
import traceback
from contextlib import contextmanager
from email.parser import Parser
from importlib.metadata import PackageNotFoundError, PathDistribution
from pathlib import Path
from tempfile import TemporaryDirectory

IMAGE = "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
CONSTRAINT = Path("/etc/sglang/constraints/torch-rocm.txt")
CONSTRAINT_SHA = "7a425435d076500cf33b24dd8cf5a1f1b24e9954a15cae03e4ed708cd1d52ef3"
SDIST_URL = "https://files.pythonhosted.org/packages/37/de/921b6491efce5c389a5ef9bbed3d2d6660005840dae488124173180859ab/torch_c_dlpack_ext-0.1.5.tar.gz"
SDIST_SHA = "d06f0357d575d22a168cc77acb9020fc4bae30968ceb6718a055dcbe92bacabe"
# Official dc0a/ca02 wheel anchors and seven source payloads installed from d06f.
ANCHORS = {
    "apache_tvm_ffi-0.1.12.dist-info/METADATA": "2248b339aa5f78915a8d4ec9a13848e99b05195edba56e438c6b6d19d3974cdb",
    "build_backend.py": "59480145c7bb8be2bb86d4dda092763953d22d550056ff03bfe704b5af204f89",
    "tilelang-0.1.15.dist-info/METADATA": "380d3486286aeb708dc4846cfdc4b438985c84199a160b21238cadf3e4e3fa19",
    "tilelang/__init__.py": "26a1ef20a9d8c1d2e3bbf378a38c66347cbb068bf414650815f5f6de50cca8b8",
    "tilelang/env.py": "afd14d73db0c671f5d794a7c9f2b4425fb5c5ccce3f41a048c68cc34ff7fb966",
    "tilelang/lib/libtilelang.so": "6b1d4267e5f46af4c21a97c5fd7dc3153423eafaad0bbc888c85a904752480ad",
    "tilelang/lib/libtvm_compiler.so": "0df6c9f287f29ab42b29819dc84fb5a58772063fad502911ae8580a47c88bdcb",
    "tilelang/lib/libtvm_runtime.so": "5fc2066d0456ca24fe7001d3358745fcf302104844e647108bca639236293f0f",
    "torch_c_dlpack_ext-0.1.5.dist-info/METADATA": (
        "7b320b8b0f9a541a2a6b20aa73088a0ec667515399a3b2725e26c4ca1eacb18d"
    ),
    "torch_c_dlpack_ext-0.1.5.dist-info/licenses/LICENSE": (
        "b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1"
    ),
    "torch_c_dlpack_ext-0.1.5.dist-info/licenses/NOTICE": (
        "f3b4f47f4a9ce94936770335500563f49ced15993dd5ac2bc97075512fbd5cd6"
    ),
    "torch_c_dlpack_ext-0.1.5.dist-info/top_level.txt": (
        "ef9cf165ab89ecd3856e35ed85d3589ac6abe2eb9efd7bc6d607396b972c344f"
    ),
    "torch_c_dlpack_ext/__init__.py": "b583dbbec872c77eea2564f4216062cf324d8b0b7c93d5e77628bf9b46e6a13e",
    "torch_c_dlpack_ext/core.py": "5804ca0f3aa3fd8feaac429f57bfb5711c7ad4d988f03c21d2eee73c11d970ff",
    "tvm_ffi/_optional_torch_c_dlpack.py": "f507e145355e4a9000d727cc53fe8fb752c51361e30d0bfaecfb46fdc232a7f0",
    "tvm_ffi/core.abi3.so": "c6152d912a1633f6eb82fc11cbe293d8f8a3305646719d8d0c1f1c886562ad74",
    "tvm_ffi/lib/libtvm_ffi.so": "bb2bb475e9a01abd69b39fdacd89db9dada4199f49b2e65da5ef1465162ed46f",
    "tvm_ffi/lib/libtvm_ffi_testing.so": "bbac9552088b630684f3bbf5bd988488b36b9809e2057c1fe77f260abef473ff",
    "tvm_ffi/registry.py": "f0378d38c7738281c21b7100d9cafa037ed421c118ede8a1484214a414d5c395",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_cache_source(name):
    """Map a recorded CPython cache path without consulting runtime cache settings."""
    path = Path(name)
    if path.parent.name != "__pycache__":
        return None
    match = re.fullmatch(r"(.+)\.cpython-[0-9]+(?:\.opt-[0-9]+)?\.pyc", path.name)
    return str(path.parent.parent / (match[1] + ".py")) if match else None


def uninstall_record_rows(record):
    """Use only path spellings whose full-text and pip line-based CSV parsing agree."""
    text = record.read_text()
    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
        pip_rows = list(csv.reader(text.splitlines()))
    except csv.Error as error:
        raise RuntimeError("Malformed uninstall RECORD") from error
    require(all(len(row) == 3 for row in rows), "Malformed uninstall RECORD")
    require(
        all(
            not any(ord(ch) < 32 or 127 <= ord(ch) <= 159 or ch in "\u2028\u2029" for ch in row[0])
            for row in rows
        )
        and all(len(row) == 3 for row in pip_rows)
        and [row[0] for row in rows] == [row[0] for row in pip_rows],
        "Unsafe uninstall RECORD parser disagreement or control character",
    )
    return rows


def verify_record(root, folder, consoles=()):
    record = folder / "RECORD"
    rows = uninstall_record_rows(record)
    require(all(len(row) == 3 for row in rows), "Malformed installed RECORD")
    names = {row[0] for row in rows}
    require(len(names) == len(rows), "Duplicate installed RECORD path")
    require(set(consoles) <= names, "Missing TVM console RECORD entry")
    for name, encoded, size in rows:
        path = root / name
        require(not path.is_symlink() and path.is_file(), "Missing or linked RECORD file: " + name)
        if name in consoles:
            require(
                path.resolve() == Path("/opt/venv/bin") / Path(name).name, "Unexpected console path"
            )
        else:
            require(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and path.resolve().is_relative_to(root),
                "RECORD path escapes site-packages",
            )
        body = path.read_bytes()
        if encoded:
            expected = (
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
            )
            require(
                encoded == expected and size == str(len(body)), "Installed RECORD mismatch: " + name
            )
        else:
            pyc = path.suffix == ".pyc" and "__pycache__" in path.parts
            if pyc:
                pyc = installed_cache_source(name) in names
            require((path == record or pyc) and size == "", "Unhashed RECORD file: " + name)
    return names


def verify_builtin_dlpack(core, torch, api):
    require(core._lib is None, f"ERROR: DLPack extension loaded its own library: {core._lib!r}")
    require(
        core.load_torch_c_dlpack_extension() is None,
        "ERROR: DLPack extension loaded a replacement builtin API",
    )
    require(
        torch.Tensor.__dlpack_c_exchange_api__ is api,
        "ERROR: DLPack extension replaced the Torch builtin C API",
    )


@contextmanager
def _source_imports():
    previous = sys.pycache_prefix, sys.dont_write_bytecode
    with TemporaryDirectory(prefix="smg-cohort-pycache-") as prefix:
        try:
            sys.pycache_prefix = prefix
            sys.dont_write_bytecode = True
            yield {**os.environ, "PYTHONPYCACHEPREFIX": prefix, "PYTHONDONTWRITEBYTECODE": "1"}
        finally:
            sys.pycache_prefix, sys.dont_write_bytecode = previous


def verify_extension_payload(root, folder, names):
    """One filesystem contract for final verification and safe repair selection."""
    source_names = {
        key for key in ANCHORS if key == "build_backend.py" or key.startswith("torch_c_dlpack_ext")
    }
    metadata_names = {
        folder.name + "/" + key
        for key in ("WHEEL", "RECORD", "INSTALLER", "REQUESTED", "direct_url.json")
    }
    pycs = {key for key in names if installed_cache_source(key) in source_names}
    require(
        source_names | metadata_names <= names <= source_names | metadata_names | pycs,
        "Unexpected or missing extension payload",
    )
    require(
        all(not (root / key).read_bytes().startswith(b"\x7fELF") for key in names),
        "Extension contains ELF payload",
    )
    direct = json.loads((folder / "direct_url.json").read_text())
    require(isinstance(direct, dict), "Malformed extension archive metadata")
    archive = direct.get("archive_info")
    require(isinstance(archive, dict), "Malformed extension archive_info")
    hashes = archive.get("hashes")
    require(isinstance(hashes, dict), "Malformed extension archive hashes")
    if "hash" in archive:
        legacy = archive["hash"]
        require(isinstance(legacy, str), "Malformed legacy archive hash")
        algorithm, separator, value = legacy.partition("=")
        require(
            separator == "="
            and re.fullmatch(r"[a-zA-Z0-9_]+", algorithm) is not None
            and re.fullmatch(r"[a-fA-F0-9]+", value) is not None
            and hashes.get(algorithm) == value,
            "Conflicting legacy archive hash",
        )
    require(
        direct.get("url") == SDIST_URL
        and hashes.get("sha256") == SDIST_SHA
        and "dir_info" not in direct
        and "vcs_info" not in direct,
        "Extension is not the official sdist build",
    )
    wheel = Parser().parsestr((folder / "WHEEL").read_text())
    require(
        wheel["Wheel-Version"] == "1.0"
        and wheel["Root-Is-Purelib"] == "true"
        and wheel.get_all("Tag") == ["py3-none-any"],
        "Expected pure Python extension wheel",
    )
    for path in (root / "torch_c_dlpack_ext", folder):
        require(not path.is_symlink(), "Linked extension directory")
        for child in path.rglob("*"):
            require(not child.is_symlink(), "Linked extension payload")
            if child.is_file():
                require(str(child.relative_to(root)) in names, "Unrecorded extension payload")
                require(
                    child.suffix not in (".so", ".dll", ".dylib", ".pth")
                    and not child.read_bytes().startswith(b"\x7fELF"),
                    "Unexpected extension native/startup payload",
                )


def verify(report):
    require(
        os.environ.get("E2E_SGLANG_GPTOSS_MXFP4_BF16", "0") != "1",
        "MXFP4 consumer requires its separately qualified extension; "
        "the cohort core does not certify or replace that guard",
    )
    require(sys.prefix == "/opt/venv", "Expected fixed-image interpreter")
    require(os.environ.get("SMG_SGLANG_ROCM_IMAGE", IMAGE) == IMAGE, "Unsupported SGLang image")
    require(
        str(CONSTRAINT) in os.environ.get("PIP_CONSTRAINT", "").split(), "Torch constraint absent"
    )
    require(
        not CONSTRAINT.is_symlink() and digest(CONSTRAINT) == CONSTRAINT_SHA,
        "Torch constraint changed",
    )
    versions = {
        name: md.version(name)
        for name in ("sglang", "torch", "apache-tvm-ffi", "tilelang", "torch-c-dlpack-ext")
    }
    expected_versions = {
        "sglang": "0.5.21",
        "torch": "2.11.0+rocm10.0.0",
        "apache-tvm-ffi": "0.1.12",
        "tilelang": "0.1.15",
        "torch-c-dlpack-ext": "0.1.5",
    }
    differences = {
        name: (versions[name], expected_versions[name])
        for name in expected_versions
        if versions[name] != expected_versions[name]
    }
    native_names = ("apache-tvm-ffi", "tilelang", "torch-c-dlpack-ext")
    observed_native = tuple(versions[name] for name in native_names)
    expected_native = tuple(expected_versions[name] for name in native_names)
    native_diagnostic = (
        "ERROR: native dependency versions (TVM, TileLang, DLPack): "
        f"observed={observed_native!r} expected={expected_native!r}; "
        if observed_native != expected_native
        else ""
    )
    require(
        versions == expected_versions,
        f"ERROR: SGLang ROCm cohort versions differ: {differences!r}; "
        f"{native_diagnostic}full cohort observed={versions!r} expected={expected_versions!r}",
    )
    root = Path(sysconfig.get_path("purelib")).resolve()
    require(root.is_relative_to(Path("/opt/venv")), "Unexpected site-packages")
    installed, record_names = {}, set()
    for name, version, package in (
        ("apache-tvm-ffi", "0.1.12", "tvm_ffi"),
        ("tilelang", "0.1.15", "tilelang"),
        ("torch-c-dlpack-ext", "0.1.5", "torch_c_dlpack_ext"),
    ):
        matches = find_native_distributions(md.distributions(), name)
        require(len(matches) == 1, "Ambiguous distribution: " + name)
        dist = matches[0]
        folder = root / (name.replace("-", "_") + "-" + version + ".dist-info")
        require(
            Path(dist.locate_file("")).resolve() == root
            and Path(dist._path).resolve() == folder
            and not folder.is_symlink(),
            "Unexpected distribution location: " + name,
        )
        consoles = (
            ("../../../bin/tvm-ffi-config", "../../../bin/tvm-ffi-stubgen")
            if name == "apache-tvm-ffi"
            else ()
        )
        names = verify_record(root, folder, consoles)
        record_names.update(names)
        if name == "torch-c-dlpack-ext":
            verify_extension_payload(root, folder, names)
        for path in (root / package, folder):
            require(not path.is_symlink(), "Linked package directory")
            for child in path.rglob("*"):
                require(not child.is_symlink(), "Linked installed payload")
                if child.is_file():
                    require(str(child.relative_to(root)) in names, "Unrecorded installed payload")
        installed[name] = {"RECORD_files": len(names), "distribution": str(folder)}
    require(set(ANCHORS) <= record_names, "Official anchor missing from installed RECORD")
    for name, anchor_sha in ANCHORS.items():
        path = root / name
        require(
            not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(root),
            "Invalid anchor path",
        )
        require(digest(path) == anchor_sha, "Official payload anchor differs: " + name)
    module_names = (
        "sglang",
        "mooncake",
        "smg_grpc_servicer",
        "tvm_ffi",
        "tilelang",
        "xgrammar",
        "torch_c_dlpack_ext.core",
    )
    families = {"torch", *(name.split(".")[0] for name in module_names)}
    preloaded = sorted(
        name
        for name in sys.modules
        if any(name == family or name.startswith(family + ".") for family in families)
    )
    require(
        not preloaded,
        "Protected modules loaded before source-only verification: " + ", ".join(preloaded),
    )
    # Validate installed cache paths above before redirecting cache_from_source.
    # The empty private prefix bypasses existing bytecode without rewriting it.
    with _source_imports() as child_environment:
        torch = importlib.import_module("torch")

        api = getattr(torch.Tensor, "__dlpack_c_exchange_api__", None)
        require(
            torch.__version__ == versions["torch"]
            and torch.version.hip
            and api is not None
            and type(api).__name__ == "PyCapsule",
            "Expected ROCm Torch builtin C API",
        )
        modules = {name: importlib.import_module(name) for name in module_names}
        for name in ("tvm_ffi", "tilelang", "torch_c_dlpack_ext.core"):
            relative = (
                "torch_c_dlpack_ext/core.py" if name.endswith(".core") else name + "/__init__.py"
            )
            require(
                Path(modules[name].__file__).resolve() == root / relative,
                "Unexpected native module origin: " + name,
            )
        require(
            Path(importlib.util.find_spec("build_backend").origin).resolve()
            == root / "build_backend.py",
            "Unexpected extension build-backend origin",
        )
        core = modules["torch_c_dlpack_ext.core"]
        verify_builtin_dlpack(core, torch, api)
        require(
            not any(
                "torch_c_dlpack" in line or "libtorch_c_dlpack_addon" in line
                for line in Path("/proc/self/maps").read_text().splitlines()
            ),
            "Addon library mapped",
        )
        source = torch.arange(12, device="cpu").reshape(3, 4)[:, ::2]
        consumer = torch.utils.dlpack.from_dlpack(source)
        require(
            consumer.device.type == "cpu"
            and consumer.dtype == source.dtype
            and consumer.shape == source.shape
            and consumer.data_ptr() == source.data_ptr()
            and consumer.stride() == source.stride(),
            "CPU DLPack alias differs",
        )
        consumer[0, 0] = 91
        require(source[0, 0].item() == 91, "CPU DLPack mutation was not shared")
        del source
        require(
            consumer.tolist() == [[91, 2], [4, 6], [8, 10]], "CPU DLPack value/lifetime differs"
        )
        check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=60,
            env=child_environment,
        )
        report.update(
            versions=versions,
            installed=installed,
            official_anchors=len(ANCHORS),
            source_origins={
                name: str(getattr(module, "__file__", None)) for name, module in modules.items()
            },
            constraint_sha256=CONSTRAINT_SHA,
            pip_check={
                "returncode": check.returncode,
                "stdout": check.stdout,
                "stderr": check.stderr,
            },
        )
        require(check.returncode == 0, "pip check failed")


def validate_uninstall_record(root, folder, package, consoles=()):
    """Reject unsafe pip-uninstall paths before considering any payload repair."""
    require(not folder.is_symlink() and folder.is_dir(), "Unsafe native distribution location")
    record = folder / "RECORD"
    require(not record.is_symlink() and record.is_file(), "Unsafe or absent uninstall RECORD")
    rows = uninstall_record_rows(record)
    require(all(len(row) == 3 for row in rows), "Malformed uninstall RECORD")
    names = [row[0] for row in rows]
    require(len(names) == len(set(names)), "Duplicate uninstall RECORD path")
    require(folder.name + "/RECORD" in names, "Uninstall RECORD does not identify itself")
    package_root = root / package
    require(not package_root.is_symlink(), "Unsafe linked native package directory")
    if package_root.exists():
        require(package_root.is_dir(), "Unsafe native package directory")
    for parent in (package_root, folder):
        require(
            not any(child.is_symlink() for child in parent.rglob("*")),
            "Unsafe linked native package payload",
        )
    for name in names:
        validate_uninstall_target(root, folder, package, name, names, consoles)
    return names


def validate_uninstall_target(root, folder, package, name, names, consoles=()):
    relative = Path(name)
    path = root / name
    if name in consoles:
        expected = Path("/opt/venv/bin") / relative.name
        require(
            path.resolve() == expected and not expected.parent.is_symlink(),
            "Unsafe uninstall console path",
        )
    else:
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and bool(relative.parts)
            and str(relative) == name
            and (
                relative.parts[0] in (package, folder.name)
                or (
                    package == "torch_c_dlpack_ext"
                    and (
                        name == "build_backend.py"
                        or (
                            installed_cache_source(name) == "build_backend.py"
                            and "build_backend.py" in names
                        )
                    )
                )
            )
            and path.resolve().is_relative_to(root),
            "Unsafe uninstall RECORD path: " + name,
        )
        require(
            not any(
                parent.is_symlink()
                for parent in path.parents
                if parent != root and parent.is_relative_to(root)
            ),
            "Unsafe linked uninstall parent",
        )
    require(
        not path.is_symlink() and (not path.exists() or path.is_file()),
        "Unsafe uninstall RECORD file: " + name,
    )


def validate_effective_uninstall_targets(root, folder, package, consoles=()):
    """Conservatively qualify pip's generated deletion paths before a selected repair."""
    names = validate_uninstall_record(root, folder, package, consoles)
    for name in names:
        source = root / name
        if not str(source).endswith(".py"):
            continue
        # pip enumerates these even when the recorded source no longer exists.
        targets = [source.parent / (source.name[:-3] + suffix) for suffix in (".pyc", ".pyo")]
        if source.exists() and os.path.splitext(source)[1] == ".py":
            targets.append(Path(importlib.util.cache_from_source(str(source))))
        for target in targets:
            try:
                relative = str(target.relative_to(root))
            except ValueError as error:
                raise RuntimeError(
                    "Unsafe derived uninstall target outside package root"
                ) from error
            # Legacy root backend caches have no qualified ownership. Absence is
            # harmless, but their parent/link boundary must still be checked.
            absent_backend_legacy = (
                package == "torch_c_dlpack_ext"
                and name == "build_backend.py"
                and relative in {"build_backend.pyc", "build_backend.pyo"}
                and not os.path.lexists(target)
            )
            if absent_backend_legacy:
                require(
                    not any(parent.is_symlink() for parent in target.parents if parent != root),
                    "Unsafe linked derived uninstall parent",
                )
                continue
            if (
                package == "torch_c_dlpack_ext"
                and installed_cache_source(relative) == "build_backend.py"
            ):
                require(
                    not os.path.lexists(target) or relative in names,
                    "Unsafe unrecorded shared backend cache",
                )
            validate_uninstall_target(root, folder, package, relative, names, consoles)


def validate_repair_script_metadata(folder):
    """Reject unqualified pip-generated deletion targets for actual repairs only."""
    # pip's importlib backend consumes these stdlib EntryPoint objects as data.
    # Never call EntryPoint.load(): entry-point target code must not execute.
    try:
        wrappers = [
            (entry.group, entry.name)
            for entry in PathDistribution(folder).entry_points
            if entry.group in ("console_scripts", "gui_scripts")
        ]
    except Exception as exc:
        raise RuntimeError("Automatic native repair refused: malformed script metadata") from exc
    # No extra wrapper deletion identity is qualified by this source contract.
    # This includes TVM: its existing two recorded wrappers remain valid on a
    # healthy cohort, but repairing script-bearing metadata needs explicit review.
    require(
        not wrappers,
        "Automatic native repair refused: unqualified console/gui wrappers: " + repr(wrappers),
    )
    scripts = folder / "scripts"
    require(not scripts.is_symlink(), "Automatic native repair refused: linked scripts metadata")
    if scripts.exists():
        require(
            scripts.is_dir() and not any(scripts.iterdir()),
            "Automatic native repair refused: unqualified dist-info/scripts metadata",
        )


def find_native_distributions(distributions, name):
    """Match both pip directory identity and declared Name before absence is inferred."""
    matches = []
    for dist in distributions:
        raw_name = dist.metadata.get("Name")
        declared = re.sub(r"[-_.]+", "-", raw_name).lower() if isinstance(raw_name, str) else None
        try:
            path = Path(dist._path)
        except (AttributeError, TypeError):
            require(declared != name, "Unsafe native distribution location")
            continue
        directory_name = None
        if path.name.endswith(".dist-info"):
            prefix, separator, _ = path.name[:-10].partition("-")
            if separator:
                directory_name = re.sub(r"[-_.]+", "-", prefix).lower()
        elif path.name.endswith(".egg-info"):
            directory_name = re.sub(r"[-_.]+", "-", path.name[:-9].split("-", 1)[0]).lower()
        if name not in (declared, directory_name):
            continue
        require(
            declared == directory_name == name and path.name.endswith(".dist-info"),
            "Unsafe native distribution identity: directory and METADATA Name disagree",
        )
        matches.append(dist)
    return matches


def native_repair_packages():
    """Select only version/RECORD/official-anchor mismatches; never install here."""
    require(
        os.environ.get("E2E_SGLANG_GPTOSS_MXFP4_BF16", "0") != "1",
        "MXFP4 consumer requires its separately qualified extension",
    )
    require(sys.prefix == "/opt/venv", "Expected fixed-image interpreter")
    root = Path(sysconfig.get_path("purelib")).resolve()
    require(root.is_relative_to(Path("/opt/venv")), "Unexpected site-packages")
    repair = []
    for name, version, package in (
        ("apache-tvm-ffi", "0.1.12", "tvm_ffi"),
        ("tilelang", "0.1.15", "tilelang"),
        ("torch-c-dlpack-ext", "0.1.5", "torch_c_dlpack_ext"),
    ):
        matches = find_native_distributions(md.distributions(), name)
        if not matches:
            # Nothing is installed for pip to uninstall. Do not infer a healthy payload.
            require(
                not (root / package).exists() and not (root / package).is_symlink(),
                "Unregistered native payload requires explicit recovery",
            )
            repair.append(name)
            continue
        require(len(matches) == 1, "Unsafe ambiguous native distribution")
        dist = matches[0]
        observed = md.version(name)
        require(
            isinstance(observed, str) and re.fullmatch(r"[A-Za-z0-9_.+!-]+", observed),
            "Unsafe native distribution version",
        )
        folder = Path(dist._path)
        require(
            Path(dist.locate_file("")).resolve() == root
            and folder.parent.resolve() == root
            and folder.resolve() == root / folder.name
            and not folder.is_symlink(),
            "Unsafe native distribution location",
        )
        consoles = (
            ("../../../bin/tvm-ffi-config", "../../../bin/tvm-ffi-stubgen")
            if name == "apache-tvm-ffi"
            else ()
        )
        # This admission is outside the repairable-integrity catch, even when
        # versions differ. Pip must never receive an unsafe installed RECORD.
        validate_uninstall_record(root, folder, package, consoles)
        needs_repair = False
        try:
            require(observed == version, "Native version differs")
            names = verify_record(root, folder, consoles)
            if name == "torch-c-dlpack-ext":
                verify_extension_payload(root, folder, names)
            for key, expected in ANCHORS.items():
                belongs = (
                    key.startswith(package + "/")
                    or key.startswith(folder.name + "/")
                    or (name == "torch-c-dlpack-ext" and key == "build_backend.py")
                )
                if belongs:
                    path = root / key
                    require(
                        key in names
                        and not path.is_symlink()
                        and path.is_file()
                        and digest(path) == expected,
                        "Native official anchor differs: " + key,
                    )
        except (PackageNotFoundError, OSError, RuntimeError, ValueError):
            needs_repair = True
        if needs_repair:
            # Fatal admission is not inside the repairable-integrity try/catch.
            # Check all pip metadata deletion inputs before append or any output.
            validate_effective_uninstall_targets(root, folder, package, consoles)
            validate_repair_script_metadata(folder)
            repair.append(name)
    return repair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?")
    parser.add_argument("--repair-plan", action="store_true")
    args = parser.parse_args()
    if args.repair_plan:
        require(args.output is None, "repair plan does not take a report path")
        for package in native_repair_packages():
            print(package)
        return 0
    report = {
        "verified": False,
        "scope": (
            "Fixed ROCm cohort key official anchors and installed RECORD consistency, "
            "excluding generated bytecode bodies. "
            "Verifier imports bypass installed bytecode caches; "
            "later engine/pytest imports are not covered. "
            "Not all official wheel bodies, GPU, model or native adapter acceptance."
        ),
    }
    try:
        verify(report)
        report["verified"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc(file=sys.stderr)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
