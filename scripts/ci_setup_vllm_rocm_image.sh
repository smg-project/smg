#!/bin/bash
# Set up an e2e job that runs inside vLLM's ROCm image (the AMD lanes).
#
# The image already ships vLLM with its ROCm stack, NIXL, UCX and MoRI, so this
# builds a venv on top of the image's packages instead of installing an engine.
# Job-specific packages live in a venv; the engine keeps its image version.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

SMG_CI_IMAGE_PACKAGE_VERSIONS="$(python3 -c '
from importlib.metadata import version
from packaging.version import Version
print(Version(version("vllm")), Version(version("grpcio-reflection")))
')"
read -r VLLM_VERSION SMG_CI_GRPC_COMPANION_VERSION <<< "$SMG_CI_IMAGE_PACKAGE_VERSIONS"

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
echo "image: $(python3 -V), ROCm $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"

# Install gRPC packages from source (not PyPI) so PR changes are always tested
$RETRY 3 10 pip install "vllm==${VLLM_VERSION}" -e crates/grpc_client/python/ -e grpc_servicer/

# The image ships pytest too; without a copy in the venv, `pytest` is the image's
# script on the system interpreter, which cannot see the venv's packages.
$RETRY 3 10 pip install --ignore-installed pytest

python3 -c "import torch, vllm, smg_grpc_servicer; assert torch.version.hip; print('vllm', vllm.__version__)"

if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$PWD/.venv/bin" >> "$GITHUB_PATH"
fi

# Carry this image's companion version into later steps of this Actions job.
# Standalone callers must pass the same override to ci_install_e2e_deps.sh.
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SMG_CI_GRPC_COMPANION_VERSION=${SMG_CI_GRPC_COMPANION_VERSION}" >> "$GITHUB_ENV"
fi
