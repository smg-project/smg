#!/bin/bash
# Set up a job that runs inside SGLang's ROCm image (the AMD benchmark row).
#
# The image provides SGLang, ROCm Torch and Mooncake in /opt/venv. Keep that
# stack, align its native FFI dependencies and install SMG's gRPC packages
# in the same interpreter. A venv of the job's own would not see
# SGLang: --system-site-packages exposes the base interpreter's packages, not
# /opt/venv's. pip keeps to the PIP_CONSTRAINT with which the image pins its
# ROCm torch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

# Preserve this image's companions instead of the generic protobuf-6 fallback.
SMG_CI_GRPC_COMPANION_VERSION="$(python3 -c '
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

try:
    print(Version(version("grpcio-reflection")))
except PackageNotFoundError:
    raise SystemExit("ERROR: SGLang ROCm image lacks grpcio-reflection metadata")
')"

echo "image: $(python3 -V) at $(command -v python3)"

# This repair is scoped to the pinned SGLang 0.5.21 / ROCm 10 image cohort.
# Build the original extension sdist against this image's Torch builtin API;
# a cached or public binary wheel may have been built for a different Torch.
python3 -c '
from importlib.metadata import version
import torch

if (version("sglang") != "0.5.21"
        or version("torch") != "2.11.0+rocm10.0.0" or not torch.version.hip):
    raise SystemExit("ERROR: unsupported SGLang ROCm native dependency cohort")
api = getattr(torch.Tensor, "__dlpack_c_exchange_api__", None)
if api is None or type(api).__name__ != "PyCapsule":
    raise SystemExit("ERROR: SGLang ROCm image lacks the Torch builtin DLPack C API")
'
$RETRY 3 10 python3 -m pip install --no-cache-dir --no-build-isolation \
    --no-binary=torch-c-dlpack-ext \
    "apache-tvm-ffi==0.1.12" "tilelang==0.1.15" \
    "torch-c-dlpack-ext @ https://files.pythonhosted.org/packages/37/de/921b6491efce5c389a5ef9bbed3d2d6660005840dae488124173180859ab/torch_c_dlpack_ext-0.1.5.tar.gz#sha256=d06f0357d575d22a168cc77acb9020fc4bae30968ceb6718a055dcbe92bacabe"

# Install gRPC packages from source (not PyPI) so PR changes are always tested
$RETRY 3 10 python3 -m pip install -e crates/grpc_client/python/ -e grpc_servicer/

python3 -c '
from importlib.metadata import version
import torch, sglang, mooncake, smg_grpc_servicer
assert torch.version.hip
api = torch.Tensor.__dlpack_c_exchange_api__
import tvm_ffi, tilelang
from torch_c_dlpack_ext import core
assert core._lib is None and api is torch.Tensor.__dlpack_c_exchange_api__
assert tuple(version(name) for name in (
    "apache-tvm-ffi", "tilelang", "torch-c-dlpack-ext"
)) == ("0.1.12", "0.1.15", "0.1.5")
print("sglang", sglang.__version__, "HIP", torch.version.hip)
'
python3 -m pip check

# Carry the observed image version into the next Actions step.
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SMG_CI_GRPC_COMPANION_VERSION=${SMG_CI_GRPC_COMPANION_VERSION}" >> "$GITHUB_ENV"
fi
