#!/bin/bash
# Install vLLM for CI on AMD GPUs (ROCm), the counterpart of ci_install_vllm.sh.
#
# vLLM publishes ROCm wheels at wheels.vllm.ai/rocm/<version>/<variant>. They
# need Python 3.12 and glibc >= 2.34, and the index also carries the matching
# torch, triton, amd-aiter and amdsmi. The host still needs the ROCm release
# the variant names: amdsmi loads the system's libamd_smi when torch imports.
# PyPI's `vllm` is the CUDA build; uv prefers the extra index, so the ROCm wheel
# wins. The NIXL and FlashInfer steps of the CUDA script are CUDA-only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

bash "${SCRIPT_DIR}/ci_ensure_python_headers.sh"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    $RETRY 3 5 bash -c 'set -o pipefail; curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "Using uv version: $(uv --version)"

# Same engine version as the CUDA lanes (scripts/ci_install_vllm.sh), so an
# AMD leg and its H100 counterpart differ only in hardware.
VLLM_VERSION="${VLLM_VERSION:-0.27.1}"
VLLM_ROCM_VARIANT="${VLLM_ROCM_VARIANT:-rocm723}"

# rocm723 -> 7.2: fail here, with the reason, rather than at the first import.
want_rocm="${VLLM_ROCM_VARIANT:4:1}.${VLLM_ROCM_VARIANT:5:1}"
have_rocm="$(cat "${ROCM_PATH:-/opt/rocm}/.info/version" 2>/dev/null || echo none)"
if [[ "$have_rocm" != "$want_rocm".* ]]; then
    echo "ERROR: vLLM's ${VLLM_ROCM_VARIANT} wheels need ROCm ${want_rocm} on this host," \
        "found ${have_rocm} at ${ROCM_PATH:-/opt/rocm}" >&2
    exit 1
fi

echo "Installing vLLM ${VLLM_VERSION} (${VLLM_ROCM_VARIANT}) on ROCm ${have_rocm}..."
$RETRY 3 10 uv pip install "vllm==${VLLM_VERSION}" \
    --extra-index-url "https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}"

# Import canary: fail here if the CUDA wheel slipped in (e.g. a Python other
# than 3.12, for which there is no ROCm wheel) or the ROCm stack can't load.
python3 -c "import torch, vllm; assert torch.version.hip, 'torch is not a ROCm build'; print('vllm', vllm.__version__, 'hip', torch.version.hip)"

echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
$RETRY 3 10 uv pip install -e crates/grpc_client/python/
$RETRY 3 10 uv pip install -e grpc_servicer/

echo "vLLM (ROCm) installation complete"
