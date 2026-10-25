#!/bin/bash
# Fixed-image E2E setup; the generic image helper remains usable by benchmarks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
if [ "${SMG_SGLANG_ROCM_IMAGE-}" != "$IMAGE" ]; then
    echo "unsupported E2E SGLang image" >&2
    exit 1
fi

# These job-local compatibility backports preserve the pinned image's Torch.
python3 "$SCRIPT_DIR/ci_repair_sglang_rocm_native.py"
# Keep the original normal RPC installation and image imports unchanged.
bash "$SCRIPT_DIR/ci_setup_sglang_rocm_image.sh"

# A fresh interpreter loads the normal editable .pth entries installed above.
# This checks CPU import/source provenance, not device availability or a model.
python3 - "$SCRIPT_DIR/.." <<'PYTHON'
import importlib
import json
from pathlib import Path
import sys

import mooncake
import sglang
import smg_grpc_proto
import smg_grpc_servicer
import torch

root = Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve() == Path("/opt/venv"), "expected image interpreter"
assert torch.version.hip, "expected ROCm Torch"
expected = {
    smg_grpc_proto: root / "crates/grpc_client/python/smg_grpc_proto/__init__.py",
    smg_grpc_servicer: root / "grpc_servicer/smg_grpc_servicer/__init__.py",
    sglang: Path("/sgl-workspace/sglang/python/sglang/__init__.py"),
}
for module, path in expected.items():
    assert Path(module.__file__).resolve() == path, f"unexpected source: {module.__name__}"
# Import all generated companions against the normally resolved runtime.
modules = {}
for name in ("common", "mlx_engine", "sglang_encoder", "sglang_scheduler",
             "tokenspeed_encoder", "tokenspeed_scheduler", "trtllm_service", "vllm_engine"):
    for suffix in ("_pb2", "_pb2_grpc"):
        module = importlib.import_module(f"smg_grpc_proto.generated.{name}{suffix}")
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(expected[smg_grpc_proto].parent / "generated")
        modules[module.__name__] = str(path)
print(json.dumps({"python": sys.executable, "prefix": sys.prefix,
                  "sglang": sglang.__version__, "HIP": torch.version.hip,
                  "editable_sources": {m.__name__: str(p) for m, p in expected.items()},
                  "generated_modules": modules, "GPU_or_model_accepted": False}))
PYTHON
python3 -m pip check

# Check the image's public NIXL bindings only on explicit NIXL lanes.
if [ "${E2E_KV_BACKEND:-}" = "nixl" ] || [ "${E2E_SGLANG_TRANSFER_BACKEND:-}" = "nixl" ]; then
    python3 -c "import torch; from nixl._api import nixl_agent; from nixl._bindings import nixlRemoteDisconnectError"
    echo "nixl import canary OK"
fi
