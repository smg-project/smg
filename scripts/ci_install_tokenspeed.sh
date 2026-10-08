#!/bin/bash
# Install the latest published TokenSpeed nightly for CI.
# CUDA tooling supports runtime JIT; SMG gRPC packages come from this checkout.
# TOKENSPEED_BUILD_ONLY=1 prepares the carrier image without per-PR SMG glue.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

# sudo is absent when this runs as root inside `docker build`; degrade to
# running the commands directly.
if command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Keep the carrier stamp compatible with the image cache tooling. The ref
# identifies the carrier, while pip selects the latest published nightly.
TOKENSPEED_REF="${TOKENSPEED_REF:-$(tr -d '[:space:]' < "${REPO_ROOT}/.github/versions/tokenspeed.ref")}"
TOKENSPEED_PREBUILT_STAMP="${TOKENSPEED_PREBUILT_STAMP:-/opt/smg-ci/tokenspeed.ref}"

# Install uv for faster package management (mirrors ci_install_sglang.sh).
# The SMG glue below uses it.
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    $RETRY 3 5 bash -c 'set -o pipefail; curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

setup_cuda_env() {
    # ── CUDA runtime setup ─────────────────────────────────────────────────
    # k8s-runner-gpu ships the NVIDIA driver + CUDA runtime libs but not the
    # SDK (nvcc, headers). Install them on demand — same approach as
    # ``ci_install_sglang.sh``. On the prebuilt image the toolkit is already
    # baked in, so this only resolves and exports the env.
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    if [ ! -x "${CUDA_HOME}/bin/nvcc" ] && [ ! -x "/usr/local/cuda-13.0/bin/nvcc" ]; then
        echo "Installing CUDA toolkit (nvcc not found)..."
        $RETRY 3 10 curl -fsSL -o /tmp/cuda-keyring.deb \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        $SUDO dpkg -i /tmp/cuda-keyring.deb
        rm /tmp/cuda-keyring.deb
        bash "${SCRIPT_DIR}/ci_apt_mirror.sh"
        $RETRY 3 10 $SUDO apt-get update -qq
        # Install the FULL CUDA 13.0 toolkit (mirrors the proven TRT-LLM lane in
        # ci_install_trtllm.sh) so the system headers -- which the kernel build
        # compiles against -- are a complete, self-consistent 13.0.88 set matching
        # the system nvcc.
        $RETRY 3 10 $SUDO apt-get install -y cuda-toolkit-13-0
    fi
    # Point CUDA_HOME at the versioned toolkit dir directly (mirrors
    # ci_install_trtllm.sh). The job env sets CUDA_HOME=/usr/local/cuda, but on this
    # runner that symlink is stale/partial: its include/ has cuda_runtime.h but not
    # crt/host_runtime.h, so the kernel's host-stub compile falls through to torch's
    # mismatched bundled crt and dies with "'__cudaLaunch' was not declared". The
    # apt-installed /usr/local/cuda-13.0 is complete (ships cuda-crt-13-0).
    if [ -x "/usr/local/cuda-13.0/bin/nvcc" ]; then
        CUDA_HOME="/usr/local/cuda-13.0"
    fi
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
    echo "Using CUDA_HOME=${CUDA_HOME} ($(${CUDA_HOME}/bin/nvcc --version | tail -1))"
    # The kernel's launch stubs need this exact header from the system toolkit; if
    # it's missing the build falls through to torch's bundled cu13 crt and fails.
    if [ -f "${CUDA_HOME}/include/crt/host_runtime.h" ]; then
        echo "system crt/host_runtime.h: present under CUDA_HOME"
    else
        echo "WARNING: ${CUDA_HOME}/include/crt/host_runtime.h is MISSING" >&2
    fi
    # Torch's JIT cpp_extension builder compiles some TokenSpeed runtime extensions
    # (e.g. ``tokenspeed_hostfunc_ext``) with plain g++ and doesn't pass
    # ``-I$CUDA_HOME/include``; expose the system CUDA headers via CPATH so those
    # g++ compiles find them (CUDA 13 keeps CCCL under ``include/cccl``).
    local _cuda_inc="${CUDA_HOME}/include:${CUDA_HOME}/include/cccl"
    export CPATH="${_cuda_inc}${CPATH:+:$CPATH}"
    export CPLUS_INCLUDE_PATH="${_cuda_inc}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export C_INCLUDE_PATH="${_cuda_inc}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
}

ensure_rdma_libs() {
    # ── RDMA runtime libraries ─────────────────────────────────────────────
    # The EPD lane moves embeddings over Mooncake, whose native extension
    # dlopens libibverbs/libnuma at import time whatever the transport is.
    # Without them the encode worker dies during startup with
    # "libibverbs.so.1: cannot open shared object file", which TokenSpeed
    # reports as a generic "please install mooncake".
    #
    # Like the CUDA toolkit and Python headers above, these belong to the
    # runner and are not part of the prebuilt payload: the package install only
    # ever pulled them in as a transitive dependency of libopenmpi-dev, so the
    # prebuilt fast path leaves the runner without them. Install them
    # explicitly on both paths, same set as ci_install_vllm.sh.
    if ldconfig -p 2> /dev/null | grep -q 'libibverbs\.so\.1'; then
        echo "RDMA libraries: libibverbs.so.1 present"
        return
    fi

    echo "libibverbs.so.1 not found; installing RDMA runtime libraries"
    if ! command -v apt-get &> /dev/null; then
        echo "ERROR: no apt-get to install the RDMA runtime libraries with" >&2
        exit 1
    fi
    export DEBIAN_FRONTEND=noninteractive
    bash "${SCRIPT_DIR}/ci_apt_mirror.sh"
    $RETRY 3 10 $SUDO apt-get update -qq
    $RETRY 3 10 $SUDO apt-get install -y --no-install-recommends libnuma1 libibverbs1 ibverbs-providers
}

install_tokenspeed() {
    # Reuse downloaded wheels across ephemeral runner jobs when available.
    if [ "${TOKENSPEED_BUILD_ONLY:-0}" != "1" ]; then
        cache="${PIP_CACHE_DIR:-/models/.ci-cache/pip}"
        if mkdir -p "$cache" 2>/dev/null && [ -w "$cache" ]; then
            export PIP_CACHE_DIR="$cache"
        fi
    fi
    requirement="tokenspeed"
    if [ -n "${TOKENSPEED_VERSION:-}" ]; then
        requirement="tokenspeed==${TOKENSPEED_VERSION}"
    fi
    $RETRY 3 10 python3 -m pip install --upgrade "$requirement" \
        --extra-index-url https://lightseek.org/whl/nightly
    python3 - <<'PY'
import os
import re
from importlib import metadata

version = metadata.version("tokenspeed")
print(f"Installed tokenspeed=={version}", flush=True)
nightly = re.fullmatch(r"\d+\.\d+\.\d+\.post(\d{8})", version)
if nightly is None:
    raise RuntimeError(f"Expected a dated TokenSpeed nightly, installed {version}")
expected = os.environ.get("TOKENSPEED_VERSION")
if expected and version != expected:
    raise RuntimeError(f"Expected tokenspeed=={expected}, installed {version}")
print(f"TokenSpeed nightly date: {nightly.group(1)}", flush=True)
PY
    $RETRY 3 10 python3 "${SCRIPT_DIR}/ci_install_flashinfer.py"
}

persist_ci_env() {
    # ── Persist env to subsequent CI steps ─────────────────────────────────
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "CUDA_HOME=$CUDA_HOME" >> "$GITHUB_ENV"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
        # See note in setup_cuda_env: needed so torch's JIT C++ extension builder
        # sees CUDA headers when it bypasses nvcc for .cpp sources.
        echo "CPATH=$CPATH" >> "$GITHUB_ENV"
        echo "CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH" >> "$GITHUB_ENV"
        echo "C_INCLUDE_PATH=$C_INCLUDE_PATH" >> "$GITHUB_ENV"
    fi
    if [ -n "${GITHUB_PATH:-}" ]; then
        # Make ``nvcc`` discoverable to downstream steps (pytest spawns the
        # worker which may trigger CUDA extension builds).
        echo "$CUDA_HOME/bin" >> "$GITHUB_PATH"
    fi
}

install_smg_glue() {
    # ── smg gRPC packages (same as other engines: from source so PR changes land) ─
    cd "$REPO_ROOT"
    echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
    # TokenSpeed's engine package pins its own builds of these modules
    # (tokenspeed-smg-grpc-proto / tokenspeed-smg-grpc-servicer). Those dists
    # install the same smg_grpc_proto / smg_grpc_servicer import paths into
    # site-packages, which shadow the editable installs below — the worker
    # would then serve stale proto descriptors ("Method not found!" for any
    # RPC added in the PR). Drop them first; the source installs replace them.
    uv pip uninstall tokenspeed-smg-grpc-proto tokenspeed-smg-grpc-servicer
    $RETRY 3 10 uv pip install -e crates/grpc_client/python/
    $RETRY 3 10 uv pip install -e grpc_servicer/
}

# ── Main ───────────────────────────────────────────────────────────────────

setup_cuda_env
# Python dev headers: Triton compiles against them at runtime, and the baked
# venv skipped the apt repair that used to provide them by accident. Shared
# with the other engine install scripts (vllm / sglang / trtllm).
bash "${SCRIPT_DIR}/ci_ensure_python_headers.sh"
ensure_rdma_libs

# Upgrade even when the carrier already contains TokenSpeed.
install_tokenspeed

if [ "${TOKENSPEED_BUILD_ONLY:-0}" = "1" ]; then
    # Image build: keep the carrier stamp for cache validation and
    # stop before the per-PR glue.
    mkdir -p "$(dirname "$TOKENSPEED_PREBUILT_STAMP")"
    printf '%s\n' "$TOKENSPEED_REF" > "$TOKENSPEED_PREBUILT_STAMP"
    echo "TokenSpeed build-only install complete (stamp: ${TOKENSPEED_PREBUILT_STAMP})"
    exit 0
fi

persist_ci_env
install_smg_glue

# ── Cutlass/quack provenance (diagnostic) ───────────────────────────────────
# TokenSpeed pins a compatible Cutlass DSL 4.6.0 / quack >=0.6.1 pair. Surface
# exactly what loads on each runner so future pin bumps remain diagnosable.
echo "=== Cutlass/quack provenance ==="
uv pip show nvidia-cutlass-dsl quack-kernels 2>/dev/null \
    | grep -iE "^(Name|Version|Location):" || true
python3 -c "
import cutlass, quack
print('import cutlass  ->', cutlass.__file__)
print('cutlass version ->', getattr(cutlass, '__version__', '?'))
print('import quack    ->', quack.__file__)
print('quack version   ->', getattr(quack, '__version__', '?'))
" || true

# ── Verification ──────────────────────────────────────────────────────────
echo "=== TokenSpeed verification ==="
python3 -c "from tokenspeed.runtime.engine.async_llm import AsyncLLM; \
    print('AsyncLLM bases:', [b.__name__ for b in AsyncLLM.__bases__])"
python3 -c "from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer; \
    print('gRPC servicer: importable')"
python3 -c "from smg_grpc_servicer.tokenspeed.encoder_servicer import _lazy_encode_request; \
    print('EncodeRequest:', _lazy_encode_request())"
# Prove Mooncake's native extension loads here rather than 20 minutes later
# inside the EPD lane, where TokenSpeed reduces the dlopen failure to a
# generic "please install mooncake". Lanes without the package skip it.
python3 -c "
import importlib.util

if importlib.util.find_spec('mooncake') is None:
    print('mooncake: not installed, skipping')
else:
    import torch  # bundled CUDA libraries must load first
    from mooncake.engine import TransferEngine

    print('mooncake TransferEngine: importable')
"
python3 -c "
import pathlib
import smg_grpc_proto
import smg_grpc_servicer

repo = pathlib.Path.cwd().resolve()
paths = [pathlib.Path(m.__file__).resolve() for m in (smg_grpc_proto, smg_grpc_servicer)]
shadowed = [str(p) for p in paths if repo not in p.parents]
assert not shadowed, f'smg gRPC modules shadowed by site-packages copies: {shadowed}'
print('smg gRPC modules resolve to repo source: OK')
"

echo "TokenSpeed installation complete"
