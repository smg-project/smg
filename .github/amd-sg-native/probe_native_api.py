"""Read the installed image and run CPU DLPack controls without a model."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import traceback
from pathlib import Path

CONSTRAINT = Path("/etc/sglang/constraints/torch-rocm.txt")
CONSTRAINT_SHA = "7a425435d076500cf33b24dd8cf5a1f1b24e9954a15cae03e4ed708cd1d52ef3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def device_guard():
    require(not any(Path(p).exists() for p in ("/dev/kfd", "/dev/dri")), "GPU devices mapped")


def constraints():
    data = CONSTRAINT.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(digest == CONSTRAINT_SHA, "Torch constraints differ from pinned image")
    return {"sha256": digest, "text": data.decode()}


def roundtrip(torch, producer):
    tensor = torch.arange(6, dtype=torch.int64, device="cpu")
    other = torch.utils.dlpack.from_dlpack(producer(tensor))
    require(other.device.type == "cpu" and other.dtype == tensor.dtype, "CPU dtype/device")
    require(torch.equal(tensor, other) and other.data_ptr() == tensor.data_ptr(), "alias/data")
    other[0] = 17
    require(tensor[0].item() == 17, "DLPack mutation not shared")
    return {"values": tensor.tolist(), "shape": list(other.shape), "shared_storage": True}


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    result = output / "result.json"
    require(not result.exists(), "refuse to overwrite evidence")
    record = {
        "status": "incomplete",
        "checks": {},
        "identity": {
            k: os.environ.get(k)
            for k in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "RUNNER_NAME")
        },
        "scope": "Installed image imports/native API and CPU DLPack only; no model or GPU proof",
        "xgrammar_grammar_matcher": "not_executed: this bounded probe checks imports only",
    }

    def save():
        result.write_text(json.dumps(record, indent=2) + "\n")

    def check(name, operation):
        record["checks"][name] = {"status": "running"}
        save()
        try:
            value = operation()
            record["checks"][name] = {"status": "passed", "result": value}
            return value
        except Exception as error:
            record["checks"][name] = {
                "status": "failed",
                "error": type(error).__name__,
                "reason": str(error),
                "traceback": traceback.format_exc()[-4000:],
            }
        finally:
            save()

    check("device_guard", device_guard)
    check("constraints_before", constraints)
    if any(c["status"] == "failed" for c in record["checks"].values()):
        record["status"] = "failed"
        save()
        return 1
    modules = {}

    def installed(name):
        module = importlib.import_module(name)
        modules[name] = module
        distribution = "apache-tvm-ffi" if name == "tvm_ffi" else name
        version = importlib.metadata.version(distribution)
        if name == "torch":
            require(version == "2.11.0+rocm10.0.0", "Installed Torch differs from image contract")
        return {"version": version, "origin": str(Path(module.__file__).resolve())}

    check("import_torch", lambda: installed("torch"))
    torch = modules.get("torch")
    if torch is not None:
        record["cuda_initialized_before"] = torch.cuda.is_initialized()

        def native_api():
            attrs = {
                name: hasattr(torch.Tensor, name)
                for name in ("__dlpack_c_exchange_api__", "__c_dlpack_exchange_api__")
            }
            record["native_tensor_api"] = attrs
            require(any(attrs.values()), "No native Tensor DLPack exchange API")
            require(not record["cuda_initialized_before"], "CUDA initialized before CPU probe")
            return attrs

        check("native_tensor_api", native_api)
        check("torch_CPU_roundtrip", lambda: roundtrip(torch, lambda tensor: tensor))
    for name in ("tvm_ffi", "tilelang", "xgrammar"):
        check("import_" + name, lambda name=name: installed(name))
    if torch is not None and callable(getattr(modules.get("tvm_ffi"), "from_dlpack", None)):
        check("FFI_CPU_roundtrip", lambda: roundtrip(torch, modules["tvm_ffi"].from_dlpack))
    else:
        record["checks"]["FFI_CPU_roundtrip"] = {
            "status": "not_executed",
            "reason": "imports failed or from_dlpack interface absent",
        }
    check("constraints_after", constraints)
    if torch is not None:
        record["cuda_initialized_after"] = torch.cuda.is_initialized()
        check(
            "CUDA_remains_uninitialized",
            lambda: require(not record["cuda_initialized_after"], "CUDA initialized by CPU probe"),
        )
    record["status"] = (
        "passed" if all(c["status"] == "passed" for c in record["checks"].values()) else "failed"
    )
    save()
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
