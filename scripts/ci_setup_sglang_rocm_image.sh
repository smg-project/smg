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
fi
