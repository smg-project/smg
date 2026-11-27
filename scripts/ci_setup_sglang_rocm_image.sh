#!/bin/bash
# Set up a job that runs inside SGLang's ROCm image (the AMD benchmark row).
#
# The image ships SGLang with its ROCm stack and Mooncake in the venv its
# python3 runs from (/opt/venv), so this installs SMG's gRPC packages there
# instead of installing an engine. A venv of the job's own would not see
# SGLang: --system-site-packages exposes the base interpreter's packages, not
# /opt/venv's. pip keeps to the PIP_CONSTRAINT with which the image pins its
# ROCm torch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

# The v0.5.21 ROCm image builds NIXL/UCX itself. Its bash.bashrc adds UCX
# to the loader path, but Actions' non-interactive shell does not read it.
# Preserve the image's ROCm build; installing PyPI nixl would add CUDA wheels.
if [ "${E2E_KV_BACKEND:-${E2E_SGLANG_TRANSFER_BACKEND:-}}" = nixl ]; then
    export LD_LIBRARY_PATH="/opt/ucx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    python3 - <<'PY'
from importlib.util import find_spec
from pathlib import Path

import torch

if not torch.version.hip:
    raise SystemExit("ERROR: SGLang NIXL requires the ROCm image's HIP torch")

import nixl

# The pinned ROCm Dockerfile aliases nixl to its source-built nixl_rocm
# package. Check that alias before accepting imports from a CUDA package.
backend = find_spec("nixl_rocm")
if backend is None or backend.origin is None:
    raise SystemExit("ERROR: SGLang ROCm image lacks its nixl_rocm backend")
if not nixl.__file__ or Path(nixl.__file__).resolve() != Path(backend.origin).resolve():
    raise SystemExit("ERROR: nixl does not resolve to the image's nixl_rocm backend")

# Exact API used by v0.5.21's disaggregation/nixl/conn.py. Import only: the
# original e2e cases, not this preflight, exercise UCX and GPU KV transfers.
from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t
from nixl._bindings import nixlBackendError, nixlCancelledError, nixlRemoteDisconnectError

print("SGLang ROCm NIXL import canary OK:", Path(nixl.__file__).resolve())
PY
fi

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

# Install gRPC packages from source (not PyPI) so PR changes are always tested
$RETRY 3 10 python3 -m pip install -e crates/grpc_client/python/ -e grpc_servicer/

python3 -c "import torch, sglang, mooncake, smg_grpc_servicer; assert torch.version.hip; print('sglang', sglang.__version__, 'HIP', torch.version.hip)"

# Carry the observed image version into the next Actions step.
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SMG_CI_GRPC_COMPANION_VERSION=${SMG_CI_GRPC_COMPANION_VERSION}" >> "$GITHUB_ENV"
    if [ "${E2E_KV_BACKEND:-${E2E_SGLANG_TRANSFER_BACKEND:-}}" = nixl ]; then
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> "$GITHUB_ENV"
    fi
fi
