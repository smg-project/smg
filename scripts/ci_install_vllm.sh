#!/bin/bash
# Install vLLM with flash-attn for CI
# Handles CUDA toolkit setup and flash-attn compilation
# Uses uv for faster package installation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"
source "${SCRIPT_DIR}/ci_prepared_backend_env.sh"

# A CPU builder must supply the target driver profile for uv's auto selection.
# Keep its cross-index fallback: some auxiliary wheels exist only on CPU.
if [ "${SMG_BUILD_PREPARED_ENV:-0}" = 1 ] && [ -z "${UV_CUDA_DRIVER_VERSION:-}" ]; then
    echo "ERROR: set UV_CUDA_DRIVER_VERSION to the target runner driver version for a CPU image build" >&2
    exit 1
fi

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# CPython dev headers: Triton (and torch's cpp_extension) compile against them
# at engine startup. Fail here, not 20 minutes later inside a JIT build. Engine
# lanes only -- CPU lanes (wheel builds) never need them.
bash "${SCRIPT_DIR}/ci_ensure_python_headers.sh"

# Install uv for faster package management (10-100x faster than pip)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    $RETRY 3 5 bash -c 'set -o pipefail; curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv version: $(uv --version)"

PREPARED_ENV=0
if smg_prepared_env_matches vllm "${VIRTUAL_ENV:-}"; then
    PREPARED_ENV=1
    echo "Using prepared vLLM dependencies"
else
    # Pinned to the release-matrix default so the CI lane tests the same engine
    # version the images ship (and stays deterministic like the sglang/trtllm
    # installs). 0.31.0 keeps the properties the lane depends on: torchcodec is
    # guaranteed (the import canary below deliberately validates it) and fastapi
    # is capped <0.137 (0.137 breaks the prometheus-fastapi-instrumentator health
    # route; verified in 0.31.0's dependency constraints).
    # --torch-backend=auto matches the torch CUDA variant to the pod's driver;
    # a CPU builder supplies the same driver through UV_CUDA_DRIVER_VERSION.
    echo "Installing vLLM..."
    $RETRY 3 10 uv pip install "vllm==0.31.0" --torch-backend=auto

    # vLLM >=0.25 eagerly imports torchcodec, which dlopens the FFmpeg shared
    # libraries (libavutil/libavcodec/libavformat/...) at import time. The runner
    # image ships none, so every worker dies importing vllm. Install distro FFmpeg
    # (the metapackage pulls the matching libav* sonames; torchcodec supports
    # FFmpeg 4-7). This step is unconditional, so refresh apt lists first.
    echo "Installing FFmpeg for torchcodec..."
    bash "${SCRIPT_DIR}/ci_apt_mirror.sh"
    if [ "$(id -u)" != 0 ] && command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi
    $RETRY 3 10 $SUDO apt-get update
    $RETRY 3 10 $SUDO apt-get install -y --no-install-recommends ffmpeg

    # NIXL for vLLM PD disaggregation. The bare metapackage pulls both cu12 and
    # cu13 backends, so install the top-level shim alone, then the backend
    # matching torch's CUDA (same normalization as vLLM's own CI).
    echo "Installing nixl..."
    CUDA_MAJOR=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])")
    $RETRY 3 10 uv pip install --no-deps "nixl>=1.2.0"
    $RETRY 3 10 uv pip install "nixl-cu${CUDA_MAJOR}>=1.2.0"

    # Remove nixl_ep (MoE all-to-all, unused in CI): vLLM imports it eagerly when
    # present, tying every worker startup to its extra native deps
    SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
    rm -rf "${SITE_PACKAGES}/nixl_ep"
fi

# Import canary: fail here (not mid-e2e) if the nixl install is broken
# (torch first so its bundled CUDA libraries are loaded)
python3 -c "import torch, nixl"
echo "nixl import canary OK"

# Import canary: fail here (not mid-e2e) if vLLM's eager torchcodec import
# can't find the FFmpeg shared libs installed above (torch first so its
# bundled CUDA libraries are loaded)
python3 -c "import torch, torchcodec, vllm"
echo "vllm/torchcodec import canary OK"

# Mooncake transfer engine, only where a lane runs MooncakeConnector PD
# workers (its own backend, the lane-wide E2E_KV_BACKEND, or an extra backend
# for a fleet that mixes transports) so a broken wheel cannot fail the
# unrelated vLLM jobs
if [ "${E2E_VLLM_KV_BACKEND:-nixl}" = "mooncake" ] \
    || [ "${E2E_KV_BACKEND:-}" = "mooncake" ] \
    || [[ ",${E2E_VLLM_EXTRA_KV_BACKENDS:-}," == *",mooncake,"* ]] \
    || [ "${SMG_BUILD_PREPARED_ENV:-0}" = 1 ]; then
    if [ "$PREPARED_ENV" = 0 ]; then
        # Mooncake's native extension links libibverbs/libnuma at load time even
        # when the transfer protocol is tcp — without these the import fails with
        # "libibverbs.so.1: cannot open shared object file".
        echo "Installing mooncake system dependencies..."
        $RETRY 3 10 $SUDO apt-get install -y --no-install-recommends libnuma1 libibverbs1 ibverbs-providers

        # The cuda13 wheel variant matches vLLM's cu130 torch stack, so no
        # libcudart.so.12 shim is needed (torch's bundled CUDA 13 runtime
        # satisfies it once torch is imported first). Pinned — floating mooncake
        # resolves have broken CI before.
        echo "Installing mooncake-transfer-engine (cuda13)..."
        $RETRY 3 10 uv pip install "mooncake-transfer-engine-cuda13==0.3.11.post1"
    fi

    # TransferEngine links the GPU driver's libcuda.so.1, absent on CPU builders.
    # GPU jobs still fail here if the install is broken; vLLM swallows that
    # ImportError at module load (torch first for CUDA libs).
    if [ "${SMG_BUILD_PREPARED_ENV:-0}" = 1 ]; then
        echo "Deferring Mooncake driver import to the GPU job"
    else
        python3 -c "import torch; from mooncake.engine import TransferEngine"
        echo "mooncake import canary OK"
    fi
fi

# FlashInfer JIT cache: vLLM JIT-compiles flashinfer kernels at engine startup
# and the pods have no CUDA toolchain — install the precompiled cache instead,
# same recipe as vLLM's own Dockerfile. Shared with the nightly A/B workflows,
# which re-run it after swapping in a per-commit vLLM wheel.
if [ "$PREPARED_ENV" = 0 ]; then
    bash "${SCRIPT_DIR}/ci_install_flashinfer_jit_cache.sh"
else
    python3 -c "import torch, flashinfer"
fi

if [ "${SMG_BUILD_PREPARED_ENV:-0}" = 1 ]; then
    smg_mark_prepared_env vllm
    exit 0
fi

# Install gRPC packages from source (not PyPI) so PR changes are always tested
echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
$RETRY 3 10 uv pip install -e crates/grpc_client/python/
$RETRY 3 10 uv pip install -e grpc_servicer/

echo "vLLM installation complete"
