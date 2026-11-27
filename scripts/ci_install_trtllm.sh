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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"
source "${SCRIPT_DIR}/ci_prepared_backend_env.sh"
if [ "$(id -u)" != 0 ] && command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi

TRTLLM_VERSION="1.3.0rc24"
NCCL_VERSION_CONSTRAINT="nvidia-nccl-cu13>=2.28.9,<=2.29.2"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export DEBIAN_FRONTEND=noninteractive
$SUDO dpkg --configure -a --force-confnew 2>/dev/null || true

# CPython dev headers: Triton (and torch's cpp_extension) compile against them
# at engine startup. Fail here, not 20 minutes later inside a JIT build. Placed
# after the dpkg repair above: the guard apt-installs python3.X-dev, and this
# lane's runners can arrive with an interrupted dpkg state that only
# `dpkg --configure -a` clears.
bash "${SCRIPT_DIR}/ci_ensure_python_headers.sh"

install_trtllm_system_deps() {
    local pkgs=("$@")
    [ ${#pkgs[@]} -gt 0 ] || return 0
    local ubuntu_repo="ubuntu2404"
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        local vid
        vid="$(. /etc/os-release && printf '%s' "${VERSION_ID:-}")"
        if [ "$vid" = "22.04" ]; then ubuntu_repo="ubuntu2204"; fi
    fi
    if ! dpkg -l cuda-keyring 2>/dev/null | grep -q '^ii'; then
        echo "Setting up NVIDIA apt repository (${ubuntu_repo})..."
        $RETRY 3 10 curl -fsSL -o /tmp/cuda-keyring.deb \
            "https://developer.download.nvidia.com/compute/cuda/repos/${ubuntu_repo}/x86_64/cuda-keyring_1.1-1_all.deb"
        $SUDO dpkg -i /tmp/cuda-keyring.deb
        rm -f /tmp/cuda-keyring.deb
    fi
    bash "${SCRIPT_DIR}/ci_apt_mirror.sh"
    # A partial refresh can otherwise succeed while leaving Ubuntu packages unavailable.
    # Refresh all sources because the install needs both Ubuntu and NVIDIA packages.
    if ! $SUDO apt-get -o APT::Update::Error-Mode=any update; then
        $RETRY 2 10 bash -c '
            bash "$1" --reprobe && ${2:+$2 }apt-get -o APT::Update::Error-Mode=any update
        ' _ "${SCRIPT_DIR}/ci_apt_mirror.sh" "$SUDO"
    fi
    $RETRY 3 10 $SUDO apt-get install -y --no-install-recommends "${pkgs[@]}"
}

has_cuda13_nvcc() {
    local dir
    for dir in /usr/local/cuda-13.0 "${CUDA_HOME:-/usr/local/cuda}"; do
        if [ -x "${dir}/bin/nvcc" ] && "${dir}/bin/nvcc" --version 2>/dev/null | grep -q "release 13\."; then
            return 0
        fi
    done
    return 1
}

PREPARED_ENV=0
if smg_prepared_env_matches trtllm "${VIRTUAL_ENV:-}"; then
    PREPARED_ENV=1
    echo "Using prepared TRT-LLM dependencies"
fi

# ── Runtime system dependencies ──────────────────────────────────────────────
# Prepared runner images carry all three packages; bare pods adopting a carrier
# venv already get /usr/local/cuda-13.0 symlinked from NVMe and only need the
# tiny host shared libraries when missing.
if [ "$PREPARED_ENV" = 0 ]; then
    missing_pkgs=(libopenmpi-dev libnvinfer10)
    if ! has_cuda13_nvcc; then
        missing_pkgs+=(cuda-toolkit-13-0)
    fi
    install_trtllm_system_deps "${missing_pkgs[@]}"
else
    if ! has_cuda13_nvcc; then
        install_trtllm_system_deps cuda-toolkit-13-0
    fi
fi

# ── CUDA runtime setup ───────────────────────────────────────────────────────
if [ -x "/usr/local/cuda-13.0/bin/nvcc" ]; then
    export CUDA_HOME="/usr/local/cuda-13.0"
else
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

if [ "$PREPARED_ENV" = 0 ]; then
    # ── Install pip and NCCL runtime ─────────────────────────────────────────
    $RETRY 3 10 pip install --upgrade pip
    $RETRY 3 10 pip install --no-cache-dir "$NCCL_VERSION_CONSTRAINT"

    # ── Install TensorRT-LLM pre-release wheel from NVIDIA's index ───────────
    # PyPI only hosts the source tarball for tensorrt-llm — installing from there
    # would trigger a full source build. The pre-built linux_x86_64 wheels live on
    # https://pypi.nvidia.com, which we add as an extra index.
    #
    # The cu130 torch index is also needed so pip resolves torch 2.10+cu130
    # (cuda-bindings==13.x) instead of the default PyPI torch (cuda-bindings==12.9.4),
    # which conflicts with tensorrt-llm's cuda-python>=13 requirement.
    #
    # cuda-bindings is pinned: 13.4.1 (2026-09-10) dropped the `reserved` field
    # of cudaIpcMemHandle_t that tensorrt-llm's IPC memory setup still reads, so
    # every multi-GPU engine died at startup with an AttributeError the moment
    # the release appeared. 13.3.1 is the last version the lane ran on.
    echo "Installing tensorrt-llm==${TRTLLM_VERSION} from pypi.nvidia.com..."
    $RETRY 3 10 pip install --no-cache-dir --pre \
        --extra-index-url https://pypi.nvidia.com \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        "tensorrt-llm==${TRTLLM_VERSION}" "cuda-bindings==13.3.1"

    # typer >= 0.26 leaks click.exceptions.Exit through its main on CLI exit, so
    # every `hf` invocation (model downloads) exits 1 even on success. The
    # transformers pulled by tensorrt-llm has an unbounded typer dependency.
    $RETRY 3 10 pip install --no-cache-dir "typer<0.26"
fi

if [ "${SMG_BUILD_PREPARED_ENV:-0}" = 1 ]; then
    echo "Deferring TRT-LLM driver-dependent import canaries to the GPU job"
    smg_mark_prepared_env trtllm
    exit 0
fi

# Import canary: fail here (not 20 minutes into the lane) if the pin above
# no longer matches what tensorrt-llm's IPC path expects.
python3 -c "from cuda.bindings import runtime; runtime.cudaIpcMemHandle_t().reserved"
echo "cuda-bindings IPC handle canary OK"

# ── Setup LD_LIBRARY_PATH ────────────────────────────────────────────────────
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
NVIDIA_LIB_DIRS=$(find "$SITE_PACKAGES/nvidia" -name "lib" -type d 2>/dev/null | sort -u | paste -sd':' || true)
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

TRTLLM_LIB_DIR=$(find "$SITE_PACKAGES" -path "*/tensorrt_llm/libs" -type d 2>/dev/null | head -1 || true)
if [ -n "$TRTLLM_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TRTLLM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Bare pods adopting a carrier venv may lack host OpenMPI / TensorRT shared libs.
# Probe import first so prepared runner images and unit tests incur zero apt work.
if [ "$PREPARED_ENV" = 1 ] && ! python3 -c "import tensorrt_llm" 2>/dev/null; then
    missing_runtime=()
    ldconfig -p 2>/dev/null | grep -q 'libmpi\.so' || missing_runtime+=(libopenmpi-dev)
    ldconfig -p 2>/dev/null | grep -q 'libnvinfer\.so\.10' || missing_runtime+=(libnvinfer10)
    if [ ${#missing_runtime[@]} -gt 0 ]; then
        echo "Installing missing TRT-LLM host runtime libraries: ${missing_runtime[*]}"
        install_trtllm_system_deps "${missing_runtime[@]}"
    fi
fi

# Later CI steps run in a new shell, so retain the toolkit as well as its libraries.
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "CUDA_HOME=$CUDA_HOME" >> "$GITHUB_ENV"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
fi
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$CUDA_HOME/bin" >> "$GITHUB_PATH"
fi

# ── Verification ─────────────────────────────────────────────────────────────
echo "=== TensorRT-LLM verification ==="
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"
echo "Verifying gRPC serve command..."
python3 -m tensorrt_llm.commands.serve serve --help 2>&1 | head -20 || echo "WARNING: serve --help failed"

echo "TensorRT-LLM installation complete (from PyPI)"
