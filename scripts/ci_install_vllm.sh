#!/bin/bash
# Install vLLM with flash-attn for CI
# Handles CUDA toolkit setup and flash-attn compilation
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

# Pinned to the release-matrix default so the CI lane tests the same engine
# version the images ship (and stays deterministic like the sglang/trtllm
# installs). 0.27.1 keeps the properties the lane depends on: torchcodec is
# guaranteed (the import canary below deliberately validates it) and fastapi
# is capped <0.137 (0.137 breaks the prometheus-fastapi-instrumentator health
# route; verified in 0.27.1's dependency constraints).
# --torch-backend=auto matches the torch CUDA variant to the pod's driver.
echo "Installing vLLM..."
uv pip install "vllm==0.27.1" --torch-backend=auto

# vLLM >=0.25 eagerly imports torchcodec, which dlopens the FFmpeg shared
# libraries (libavutil/libavcodec/libavformat/...) at import time. The runner
# image ships none, so every worker dies importing vllm. Install distro FFmpeg
# (the metapackage pulls the matching libav* sonames; torchcodec supports
# FFmpeg 4-7). This step is unconditional, so refresh apt lists first.
echo "Installing FFmpeg for torchcodec..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends ffmpeg

# NIXL for vLLM PD disaggregation. The bare metapackage pulls both cu12 and
# cu13 backends, so install the top-level shim alone, then the backend
# matching torch's CUDA (same normalization as vLLM's own CI).
echo "Installing nixl..."
CUDA_MAJOR=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])")
uv pip install --no-deps "nixl>=1.2.0"
uv pip install "nixl-cu${CUDA_MAJOR}>=1.2.0"

# Remove nixl_ep (MoE all-to-all, unused in CI): vLLM imports it eagerly when
# present, tying every worker startup to its extra native deps
SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
rm -rf "${SITE_PACKAGES}/nixl_ep"

# Import canary: fail here (not mid-e2e) if the nixl install is broken
# (torch first so its bundled CUDA libraries are loaded)
python3 -c "import torch, nixl"
echo "nixl import canary OK"

# Import canary: fail here (not mid-e2e) if vLLM's eager torchcodec import
# can't find the FFmpeg shared libs installed above (torch first so its
# bundled CUDA libraries are loaded)
python3 -c "import torch, torchcodec, vllm"
echo "vllm/torchcodec import canary OK"

# Mooncake transfer engine, only on the MooncakeConnector PD leg so a broken
# wheel cannot fail the unrelated vLLM jobs
if [ "${E2E_VLLM_KV_BACKEND:-nixl}" = "mooncake" ]; then
    # Mooncake's native extension links libibverbs/libnuma at load time even
    # when the transfer protocol is tcp — without these the import fails with
    # "libibverbs.so.1: cannot open shared object file".
    echo "Installing mooncake system dependencies..."
    sudo apt-get install -y --no-install-recommends libnuma1 libibverbs1 ibverbs-providers

    # The cuda13 wheel variant matches vLLM's cu130 torch stack, so no
    # libcudart.so.12 shim is needed (torch's bundled CUDA 13 runtime
    # satisfies it once torch is imported first). Pinned — floating mooncake
    # resolves have broken CI before.
    echo "Installing mooncake-transfer-engine (cuda13)..."
    uv pip install "mooncake-transfer-engine-cuda13==0.3.11.post1"

    # Import canary: fail here (not mid-e2e) if the mooncake install is broken —
    # vLLM swallows this ImportError at module load (torch first for CUDA libs)
    python3 -c "import torch; from mooncake.engine import TransferEngine"
    echo "mooncake import canary OK"
fi

# FlashInfer JIT cache: vLLM JIT-compiles flashinfer kernels at engine startup
# and the pods have no CUDA toolchain — install the precompiled cache instead,
# same recipe as vLLM's own Dockerfile.
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
uv pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
    --index-url "https://flashinfer.ai/whl/cu${CUDA_TAG}"

# Install gRPC packages from source (not PyPI) so PR changes are always tested
echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/

echo "vLLM installation complete"
