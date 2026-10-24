#!/usr/bin/env python3
"""Apply the native compatibility fixes for the pinned AMD SGLang e2e image."""

import base64
import csv
import hashlib
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
    main()
