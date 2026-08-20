#!/bin/bash
# Install TensorRT-LLM pre-release wheel from PyPI for CI.
#
# The gRPC serve command from PR #11037 and the Harmony parser fixes (#12045,
# #12467) referenced by SMG #801 first shipped in 1.3.0rc14 (released
# 2026-05-07) and remain included in the pinned 1.3.0rc24 pre-release wheel.
# We install it directly from PyPI instead of building TensorRT-LLM from
# source, which saves ~30 min of CMake compile time per CI run. See git
# history for the previous source-build logic.
#
# Prerequisites (expected on k8s-runner-gpu nodes):
#   - NVIDIA driver 580+ (CUDA 13)
#   - CUDA 13.0 toolkit at /usr/local/cuda-13.0
#   - H100 GPUs (sm90)
#
# At runtime we use --backend pytorch, which avoids TRT engine compilation.

set -euo pipefail

TRTLLM_VERSION="1.3.0rc24"
NCCL_VERSION_CONSTRAINT="nvidia-nccl-cu13>=2.28.9,<=2.29.2"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ── Runtime system dependencies ──────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
sudo dpkg --configure -a --force-confnew 2>/dev/null || true

# Add NVIDIA apt repository if needed
if ! dpkg -l cuda-keyring 2>/dev/null | grep -q '^ii'; then
    echo "Setting up NVIDIA apt repository..."
    curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb
fi

sudo apt-get update
# Runtime deps: wheel links against CUDA 13 + TensorRT libs
sudo apt-get install -y libopenmpi-dev libnvinfer10 cuda-toolkit-13-0

# ── CUDA runtime setup ───────────────────────────────────────────────────────
if [ -d "/usr/local/cuda-13.0" ]; then
    export CUDA_HOME="/usr/local/cuda-13.0"
else
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

# ── Install pip and NCCL runtime ─────────────────────────────────────────────
pip install --upgrade pip
pip install --no-cache-dir "$NCCL_VERSION_CONSTRAINT"

# ── Install TensorRT-LLM pre-release wheel from NVIDIA's index ───────────────
# PyPI only hosts the source tarball for tensorrt-llm — installing from there
# would trigger a full source build. The pre-built linux_x86_64 wheels live on
# https://pypi.nvidia.com, which we add as an extra index.
#
# The cu130 torch index is also needed so pip resolves torch 2.10+cu130
# (cuda-bindings==13.x) instead of the default PyPI torch (cuda-bindings==12.9.4),
# which conflicts with tensorrt-llm's cuda-python>=13 requirement.
echo "Installing tensorrt-llm==${TRTLLM_VERSION} from pypi.nvidia.com..."
pip install --no-cache-dir --pre \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    "tensorrt-llm==${TRTLLM_VERSION}"

# typer >= 0.26 leaks click.exceptions.Exit through its main on CLI exit, so
# every `hf` invocation (model downloads) exits 1 even on success. The
# transformers pulled by tensorrt-llm has an unbounded typer dependency.
pip install --no-cache-dir "typer<0.26"

# ── Setup LD_LIBRARY_PATH ────────────────────────────────────────────────────
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
NVIDIA_LIB_DIRS=$(find "$SITE_PACKAGES/nvidia" -name "lib" -type d 2>/dev/null | sort -u | paste -sd':')
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

TRTLLM_LIB_DIR=$(find "$SITE_PACKAGES" -path "*/tensorrt_llm/libs" -type d 2>/dev/null | head -1)
if [ -n "$TRTLLM_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TRTLLM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Persist LD_LIBRARY_PATH for subsequent CI steps
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
fi

# ── Verification ─────────────────────────────────────────────────────────────
echo "=== TensorRT-LLM verification ==="
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"
echo "Verifying gRPC serve command..."
python3 -m tensorrt_llm.commands.serve serve --help 2>&1 | head -20 || echo "WARNING: serve --help failed"

echo "TensorRT-LLM installation complete (from PyPI)"
