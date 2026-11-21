#!/bin/bash
# Set up an e2e job that runs inside vLLM's ROCm image (the AMD lanes).
#
# The image already ships vLLM with its ROCm stack, NIXL, UCX and MoRI, so this
# builds a venv on top of the image's packages instead of installing an engine.
# The venv exists because smg-grpc-servicer needs grpcio >= 1.81.1 and the
# image's vLLM pins 1.78.0; in vLLM only the gRPC launcher imports grpc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
echo "image: $(python3 -V), ROCm $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"

# Install gRPC packages from source (not PyPI) so PR changes are always tested
$RETRY 3 10 pip install -e crates/grpc_client/python/ -e grpc_servicer/

# The image ships pytest too; without a copy in the venv, `pytest` is the image's
# script on the system interpreter, which cannot see the venv's packages.
$RETRY 3 10 pip install --ignore-installed pytest

python3 -c "import torch, vllm, smg_grpc_servicer; assert torch.version.hip; print('vllm', vllm.__version__)"

# Any explicitly requested Mooncake transport needs this dependency canary.
# Importability does not qualify HIP transfers, devices or the native cohort.
python3 - <<'PY_CANARY'
import os

names = [
    os.environ.get("E2E_VLLM_KV_BACKEND", ""),
    os.environ.get("E2E_KV_BACKEND", ""),
]
names += os.environ.get("E2E_VLLM_EXTRA_KV_BACKENDS", "").split(",")
if "mooncake" in {name.strip().lower() for name in names if name.strip()}:
    print("Checking requested Mooncake backend", flush=True)
    import torch
    from mooncake.engine import TransferEngine

    print("mooncake import canary OK")
PY_CANARY

if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$PWD/.venv/bin" >> "$GITHUB_PATH"
fi
