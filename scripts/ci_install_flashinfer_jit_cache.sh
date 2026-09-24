#!/bin/bash
# Install the flashinfer-jit-cache wheel matching the installed flashinfer-python.
#
# vLLM JIT-compiles flashinfer kernels at engine startup and the CI pods have no
# CUDA toolchain, so the precompiled cache is mandatory (same recipe as vLLM's own
# Dockerfile). flashinfer refuses to import when the two packages' versions differ,
# so this must be re-run whenever flashinfer-python changes — ci_install_vllm.sh
# calls it after the pinned vLLM install, and the nightly A/B workflows call it
# again after swapping in a per-commit vLLM main wheel (which pulls a newer
# flashinfer-python).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"

echo "Installing flashinfer-jit-cache..."
CUDA_TAG=$(python3 -c "import torch; print(torch.version.cuda.replace('.', ''))")
FLASHINFER_VERSION=$(python3 -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")
# flashinfer hosts one wheel index per CUDA tag and lags new CUDA minors
# (torch moved to +cu132 while the newest index is cu130; a missing index
# 404s and uv reports it as "package not found"). CUDA minor versions are
# ABI-compatible, so walk down to the nearest published tag in this major.
CUDA_MAJOR_FLOOR=$((CUDA_TAG / 10 * 10))
while [ "${CUDA_TAG}" -gt "${CUDA_MAJOR_FLOOR}" ] \
    && ! curl -sfo /dev/null "https://flashinfer.ai/whl/cu${CUDA_TAG}/flashinfer-jit-cache/"; do
    CUDA_TAG=$((CUDA_TAG - 1))
done
$RETRY 3 10 uv pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
    --index-url "https://flashinfer.ai/whl/cu${CUDA_TAG}"

# Import canary: flashinfer checks the two versions at import time — fail here,
# not at engine startup, if the cache wheel does not match.
python3 -c "import torch, flashinfer; print('flashinfer', flashinfer.__version__)"
echo "flashinfer-jit-cache ${FLASHINFER_VERSION} (cu${CUDA_TAG}) OK"
