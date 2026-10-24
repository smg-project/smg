#!/usr/bin/env python3
"""Apply the native compatibility fixes for the pinned AMD SGLang e2e image."""

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata as md
import io
import json
import os
import subprocess
import sys
from pathlib import Path

IMAGE = "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
CONSTRAINT = Path("/etc/sglang/constraints/torch-rocm.txt")
CONSTRAINT_SHA = "7a425435d076500cf33b24dd8cf5a1f1b24e9954a15cae03e4ed708cd1d52ef3"
WHEELS = [
    "https://files.pythonhosted.org/packages/c8/2a/bff8d73841b49f196852edd8460241d3a363e6b0d64c3a9367542658394f/apache_tvm_ffi-0.1.12-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=817af52916ca9987e019ae9c811406835c7f26c590b2a7bcfa9db0e3809f4228",
    "https://files.pythonhosted.org/packages/e2/79/a914539b4785f3e44f891aa012a886edb8bc10fe081c440981c57543ce21/torch_c_dlpack_ext-0.1.5-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl#sha256=e6f9da4bb9af70e27facc777458be62e10dbbbddda7672d16138db0553c5a524",
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patched(data, spec):
    if sha(data) == spec["fixed_sha256"]:
        return data
    require(sha(data) == spec["old_sha256"], "unrecognized source: " + spec["path"])
    old, new = spec["old"].encode(), spec["new"].encode()
    require(data.count(old) == 1, "patch context is not unique")
    fixed = data.replace(old, new)
    require(sha(fixed) == spec["fixed_sha256"], "patched source identity differs")
    return fixed


def read(path):
    with Path(path).open("rb") as stream:
        data = stream.read(512 * 1024 + 1)
    require(len(data) <= 512 * 1024, "source exceeds size limit")
    return data


def update_record(dist, registry):
    # Keep the installed wheel's file inventory truthful after the explicit Python backport.
    record = next(dist.locate_file(f) for f in dist.files if str(f).endswith(".dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(record.read_text())))
    entry = next(row for row in rows if row[0] == "tvm_ffi/registry.py")
    entry[1:] = [
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(registry).digest()).decode().rstrip("="),
        str(len(registry)),
    ]
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    record.write_text(stream.getvalue())


FFI_FILES = {
    "tvm_ffi/core.abi3.so": "48dbc71a92a0d134feaeb5f81d7e7760485e856404d1e26e776330fcc6cfea06",
    "tvm_ffi/lib/libtvm_ffi.so": "4f3e1f951aa56d2047b3353e75d6cea112f3f4ecf29fb4910b7b6ca11ca93c90",
    "tvm_ffi/lib/libtvm_ffi_testing.so": "b8edbbcd096553cdc5479df08ff4b010c1746266e0872bd7a5492fc1a5767230",
    "apache_tvm_ffi-0.1.12.dist-info/METADATA": "2248b339aa5f78915a8d4ec9a13848e99b05195edba56e438c6b6d19d3974cdb",
}


def verify_files(plan, dist):
    sources = {}
    for spec in plan:
        sources[spec["path"]] = sha(read(spec["path"]))
        require(
            sources[spec["path"]] == spec["fixed_sha256"], "native source changed: " + spec["path"]
        )
    files = {str(f): f for f in dist.files or ()}
    expected = {**FFI_FILES, "tvm_ffi/registry.py": plan[0]["fixed_sha256"]}
    identities = {}
    for name, expected_sha in expected.items():
        require(name in files, "FFI RECORD entry absent: " + name)
        entry = files[name]
        data = dist.locate_file(entry).read_bytes()
        require(sha(data) == expected_sha, "FFI installed bytes changed: " + name)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        require(
            entry.hash
            and entry.hash.mode == "sha256"
            and entry.hash.value == digest
            and entry.size == len(data),
            "FFI RECORD differs: " + name,
        )
        identities[name] = {"sha256": sha(data), "bytes": len(data)}
    return {"fixed_sources": sources, "ffi_record_files": identities}


def verify_only(output):
    report = {"verified": False, "python": sys.executable, "prefix": sys.prefix}
    code = 1
    try:
        require(os.environ.get("SMG_SGLANG_ROCM_IMAGE") == IMAGE, "unsupported SGLang image")
        require(
            str(CONSTRAINT) in os.environ.get("PIP_CONSTRAINT", "").split(),
            "Torch constraint absent",
        )
        report["constraints"] = {
            "environment": os.environ["PIP_CONSTRAINT"],
            "path": str(CONSTRAINT),
            "sha256": sha(read(CONSTRAINT)),
        }
        require(report["constraints"]["sha256"] == CONSTRAINT_SHA, "unexpected Torch constraints")
        versions = {
            "sglang": "0.5.21",
            "tilelang": "0.1.7.post3+cuda.gita55a8230",
            "apache-tvm-ffi": "0.1.12",
            "torch-c-dlpack-ext": "0.1.5",
            "torch": "2.11.0+rocm10.0.0",
        }
        report["versions"] = {name: md.version(name) for name in versions}
        require(report["versions"] == versions, "native package version changed")
        require(
            "torch-c-dlpack-ext" in md.requires("tilelang"), "addon metadata requirement changed"
        )
        plan = json.loads(Path(__file__).with_name("sglang_rocm_native_patches.json").read_text())
        report["files"] = verify_files(plan, md.distribution("apache-tvm-ffi"))
        import torch

        require(
            hasattr(torch.Tensor, "__dlpack_c_exchange_api__"), "native Torch DLPack API required"
        )
        report["torch_native_dlpack_api"] = True
        # This CLI is a fresh process; these imports do not run GPU arithmetic or a model.
        modules = [
            importlib.import_module(name)
            for name in ["tvm_ffi.registry", "tvm", "tilelang", "xgrammar"]
        ]
        report["imports"] = {m.__name__: m.__file__ for m in modules}
        require(
            sha(read(modules[0].__file__)) == plan[0]["fixed_sha256"], "loaded FFI registry changed"
        )
        report["verified"] = True
        code = 0
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(str(exc), file=sys.stderr)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return code


def main():
    require(os.environ.get("SMG_SGLANG_ROCM_IMAGE") == IMAGE, "unsupported SGLang image")
    require(
        str(CONSTRAINT) in os.environ.get("PIP_CONSTRAINT", "").split(), "Torch constraint absent"
    )
    original = CONSTRAINT.read_bytes()
    require(sha(original) == CONSTRAINT_SHA, "unexpected Torch constraints")
    import torch

    require(torch.__version__ == "2.11.0+rocm10.0.0", "unexpected Torch version")
    require(hasattr(torch.Tensor, "__dlpack_c_exchange_api__"), "native Torch DLPack API required")
    require(md.version("sglang") == "0.5.21", "unexpected SGLang version")
    require(md.version("tilelang") == "0.1.7.post3+cuda.gita55a8230", "unexpected TileLang version")
    # TileLang requires this package; its old CUDA/CPU libraries are never the ROCm execution path.
    require("torch-c-dlpack-ext" in md.requires("tilelang"), "addon metadata requirement changed")
    torch_before = {
        n: (md.version(n), md.requires(n)) for n in ["torch", "torchaudio", "torchvision"]
    }
    plan = json.loads(Path(__file__).with_name("sglang_rocm_native_patches.json").read_text())
    for spec in plan[1:]:
        patched(read(spec["path"]), spec)
    subprocess.run([sys.executable, "-m", "pip", "install", "--retries", "0", *WHEELS], check=True)
    require(CONSTRAINT.read_bytes() == original, "Torch constraints changed")
    require(
        torch_before == {n: (md.version(n), md.requires(n)) for n in torch_before},
        "Torch packages changed",
    )
    require(md.version("apache-tvm-ffi") == "0.1.12", "unexpected installed FFI version")
    require(md.version("torch-c-dlpack-ext") == "0.1.5", "unexpected installed addon version")
    changes = [(spec, patched(read(spec["path"]), spec)) for spec in plan]
    for spec, fixed in changes:
        Path(spec["path"]).write_bytes(fixed)
        require(sha(read(spec["path"])) == spec["fixed_sha256"], "installed source differs")
        print(json.dumps({"path": spec["path"], "sha256": spec["fixed_sha256"]}))
    update_record(md.distribution("apache-tvm-ffi"), Path(plan[0]["path"]).read_bytes())
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import tvm_ffi, tvm, tilelang, xgrammar; "
            "print('SGLang ROCm native dependencies imported successfully')",
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", metavar="OUTPUT_JSON")
    args = parser.parse_args()
    if args.verify_only:
        raise SystemExit(verify_only(args.verify_only))
    main()
