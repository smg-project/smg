#!/bin/bash
# Install SGLang for CI
# Uses uv for faster package installation

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install uv for faster package management (10-100x faster than pip)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv version: $(uv --version)"

# Install CUDA toolkit (nvcc) — required for SGLang JIT kernel compilation.
# SGLang >= 0.5.9 JIT-compiles CUDA kernels (RoPE, etc.) at runtime via tvm_ffi,
# which invokes nvcc. The CI runners have CUDA runtime (driver) but not the compiler.
# sglang 0.5.18 pins torch==2.13.0, whose default PyPI wheels are CUDA 13, so the
# compiler must be nvcc 13 to match the runtime headers (a 12.x nvcc is replaced).
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "${CUDA_HOME}/bin/nvcc" ] || ! "${CUDA_HOME}/bin/nvcc" --version | grep -q "release 13\."; then
    echo "Installing CUDA 13 toolkit (nvcc 13 not found at ${CUDA_HOME}/bin/nvcc)..."
    curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends cuda-nvcc-13-0 cuda-cudart-dev-13-0
    # Ensure CUDA_HOME points to the installed toolkit
    if [ ! -d "${CUDA_HOME}/bin" ] && [ -d "/usr/local/cuda-13.0/bin" ]; then
        sudo ln -sfn /usr/local/cuda-13.0 "${CUDA_HOME}"
    fi
    echo "nvcc installed: $(${CUDA_HOME}/bin/nvcc --version | tail -1)"
else
    echo "nvcc 13 already available: $(${CUDA_HOME}/bin/nvcc --version | tail -1)"
fi

# Install SGLang with all dependencies
echo "Installing SGLang..."
uv pip install --prerelease=allow "sglang[all]==0.5.18"

# Install flashinfer-jit-cache: sglang bundles flashinfer_python but only for attention ops.
# Multi-GPU models need trtllm_comm kernels (fused allreduce + layernorm) which FlashInfer
# JIT-compiles at runtime requiring nvcc. The jit-cache provides these pre-compiled.
# Version must match flashinfer_python from sglang.
FLASHINFER_VERSION=$(uv pip show flashinfer-python 2>/dev/null | grep "^Version:" | awk '{print $2}')
CU_VERSION=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "cu130")
if [ -n "$FLASHINFER_VERSION" ]; then
    # flashinfer hosts one wheel index per CUDA tag and lags new CUDA minors
    # (a missing index 404s and uv reports it as "package not found"). CUDA
    # minor versions are ABI-compatible, so walk down to the nearest
    # published tag in this major.
    CUDA_TAG=${CU_VERSION#cu}
    CUDA_MAJOR_FLOOR=$((CUDA_TAG / 10 * 10))
    while [ "${CUDA_TAG}" -gt "${CUDA_MAJOR_FLOOR}" ] \
        && ! curl -sfo /dev/null "https://flashinfer.ai/whl/cu${CUDA_TAG}/flashinfer-jit-cache/"; do
        CUDA_TAG=$((CUDA_TAG - 1))
    done
    CU_VERSION="cu${CUDA_TAG}"
    echo "Installing flashinfer-jit-cache==${FLASHINFER_VERSION} (${CU_VERSION})..."
    uv pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
        --index-url "https://flashinfer.ai/whl/${CU_VERSION}"
else
    echo "WARNING: flashinfer-python not found, skipping flashinfer-jit-cache install"
fi

# Install mooncake for SGLang PD disaggregation (KV transfer)
# Mooncake's native transfer engine requires InfiniBand/RDMA libraries at runtime.
# Package and pin track upstream sglang v0.5.18 CI on the cu13 stack
# (cuda13 wheel variant + nvrtc, since torch 2.13 defaults to CUDA 13):
# https://github.com/sgl-project/sglang/blob/v0.5.18/scripts/ci/cuda/ci_install_dependency.sh
echo "Installing mooncake system dependencies..."
sudo apt-get install -y --no-install-recommends libnuma-dev libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils
echo "Installing mooncake..."
uv pip install mooncake-transfer-engine-cuda13==0.3.12.post1 nvidia-cuda-nvrtc

# Install gRPC packages from source (not PyPI) so PR changes are always tested
echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/

echo "SGLang installation complete"
