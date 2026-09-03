#!/bin/bash
# Install TokenSpeed from source (engine + kernel + scheduler) for CI.
#
# Mirrors the upstream install pattern (see tokenspeed's docs / test/ci_system/
# install_deps.sh): one editable pip install per package, in engine →
# kernel → scheduler order. The kernel package's metadata pulls in its
# own CUDA dependencies, so we don't pre-install requirements files.
#
# Prerequisites (expected on k8s-runner-gpu nodes):
#   - NVIDIA driver 580+ (CUDA 13)
#   - CUDA 13.0 toolkit at /usr/local/cuda-13.0 or /usr/local/cuda
#   - H100 GPUs (sm90)
#
# Fast path: docker/ci-tokenspeed.Dockerfile bakes this script's source build
# (TOKENSPEED_BUILD_ONLY=1) into the CI image and stamps the built ref at
# /opt/smg-ci/tokenspeed.ref. When that stamp matches the pinned ref, the
# source build is skipped and only the per-PR SMG gRPC glue below runs —
# that part must run every job so proto/servicer changes in the PR land.
#
# Env knobs:
#   TOKENSPEED_BUILD_ONLY=1    image build: run the source build and write the
#                              stamp, skip the SMG glue + verification (the
#                              per-PR glue never belongs in the image).
#   TOKENSPEED_FORCE_SOURCE=1  ignore a matching stamp and build from source.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# sudo is absent when this runs as root inside `docker build`; degrade to
# running the commands directly.
if command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Pinned SHA from lightseekorg/tokenspeed main, single-sourced from
# .github/versions/tokenspeed.ref (also read by ci-tokenspeed-image.yml, the
# prebuilt-image resolution in pr-test-rust.yml, and
# check_engine_versions.sh). Bump the file explicitly (the engine-watch
# workflow files an issue when this drifts) rather than floating against
# ``main`` — upstream has renamed APIs before and the gRPC servicer broke
# until we caught up.
if [ -z "${TOKENSPEED_REF:-}" ]; then
    TOKENSPEED_REF="$(tr -d '[:space:]' < "${REPO_ROOT}/.github/versions/tokenspeed.ref")"
fi
if [ -z "$TOKENSPEED_REF" ]; then
    echo "ERROR: could not resolve TOKENSPEED_REF (env unset and ref file empty)" >&2
    exit 1
fi
TOKENSPEED_REPO="${TOKENSPEED_REPO:-https://github.com/lightseekorg/tokenspeed.git}"
TOKENSPEED_DIR="${TOKENSPEED_DIR:-/tmp/tokenspeed-src}"
# Stamp written by the prebuilt-image build (TOKENSPEED_BUILD_ONLY=1);
# presence + ref match means this container already carries the pinned build.
TOKENSPEED_PREBUILT_STAMP="${TOKENSPEED_PREBUILT_STAMP:-/opt/smg-ci/tokenspeed.ref}"

# Install uv for faster package management (mirrors ci_install_sglang.sh).
# Both the source build and the SMG glue below use it.
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
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
        curl -fsSL -o /tmp/cuda-keyring.deb \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        $SUDO dpkg -i /tmp/cuda-keyring.deb
        rm /tmp/cuda-keyring.deb
        $SUDO apt-get update -qq
        # Install the FULL CUDA 13.0 toolkit (mirrors the proven TRT-LLM lane in
        # ci_install_trtllm.sh) so the system headers -- which the kernel build
        # compiles against -- are a complete, self-consistent 13.0.88 set matching
        # the system nvcc.
        $SUDO apt-get install -y cuda-toolkit-13-0
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

# ── github.com authentication for the TokenSpeed fetch ──────────────────────
# TokenSpeed is the only engine installed from a git clone -- sglang, vllm and
# trtllm all come from PyPI -- so it is the only one exposed to github.com
# refusing anonymous git traffic. When the runners started doing that, this
# fetch died instantly with "could not read Username for 'https://github.com'".
#
# Authenticate it when a token is available, matching how every other place in
# this repo reaches a foreign repo (`secrets.GH_SYNC_TOKEN` for the engine
# image, `secrets.TOKENSPEED_GITHUB_TOKEN` for the whl index). The repo is
# public, so the workflow's own read-only token is enough; an unset token
# leaves the fetch anonymous so local runs keep working.
GIT_AUTH_ARGS=()
if [ -n "${TOKENSPEED_GITHUB_TOKEN:-}" ]; then
    # Same header form actions/checkout uses. Kept out of the remote URL so it
    # cannot leak through `git remote -v` or an error message.
    TOKENSPEED_AUTH_HEADER=$(printf 'x-access-token:%s' "$TOKENSPEED_GITHUB_TOKEN" | base64 -w0)
    GIT_AUTH_ARGS=(-c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${TOKENSPEED_AUTH_HEADER}")
    echo "Fetching TokenSpeed with an authenticated github.com header"
else
    echo "No TOKENSPEED_GITHUB_TOKEN set; fetching TokenSpeed anonymously"
fi

install_tokenspeed_from_source() {
    # ── Clone TokenSpeed ───────────────────────────────────────────────────
    # ``git clone --branch`` only accepts branch/tag names, not SHAs, so we
    # init+fetch+checkout instead. Works for both SHAs and refs.
    if [ ! -d "$TOKENSPEED_DIR" ]; then
        echo "Cloning TokenSpeed ${TOKENSPEED_REF} from ${TOKENSPEED_REPO}..."
        git init -q "$TOKENSPEED_DIR"
        (cd "$TOKENSPEED_DIR" \
            && git remote add origin "$TOKENSPEED_REPO" \
            && git "${GIT_AUTH_ARGS[@]}" fetch --depth 1 origin "$TOKENSPEED_REF" \
            && git checkout FETCH_HEAD)
    else
        echo "TokenSpeed clone exists at $TOKENSPEED_DIR, reusing"
        (cd "$TOKENSPEED_DIR" \
            && git "${GIT_AUTH_ARGS[@]}" fetch --depth 1 origin "$TOKENSPEED_REF" \
            && git checkout "$TOKENSPEED_REF")
    fi

    cd "$TOKENSPEED_DIR"

    # ── System dependencies (mirrors docker/Dockerfile) ────────────────────
    export DEBIAN_FRONTEND=noninteractive
    $SUDO apt-get update -qq
    $SUDO apt-get install -y --no-install-recommends libssl-dev libopenmpi-dev cmake

    # ── TokenSpeed packages ────────────────────────────────────────────────
    export MAX_JOBS="${MAX_JOBS:-16}"
    export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-9.0a 10.0a}"
    # Select the CUDA kernel backend explicitly, as TokenSpeed's own install_deps.sh
    # does on the kernel build (otherwise the native build path can differ).
    export TOKENSPEED_KERNEL_BACKEND="${TOKENSPEED_KERNEL_BACKEND:-cuda}"

    # The kernel's torch cpp_extension build must link a torch built for CUDA 13.
    # TokenSpeed's CI runs on a cu130 Docker base image that already ships it; the
    # generic k8s runner does not, so pip/uv would pull the default PyPI torch
    # (CUDA 12.x). That drops nvidia-cuda-runtime-cu12's own crt/host_runtime.h on
    # the include path, and nvcc 13's cudafe++ then generates a host stub that fails
    # to compile against those cu12 headers: "'__cudaLaunch' was not declared".
    # Point pip/uv at the cu130 wheel index (mirrors install_deps.sh line 118) so
    # every install below resolves the CUDA-13 torch + nvidia deps.
    export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    # Match pip's cross-index best-version semantics (what upstream's pip-based
    # install_deps.sh relies on). uv's default first-index strategy pins each
    # package to the first index carrying it, so the cu130 index's stale
    # ``packaging`` (<=24.1) would block flashinfer-python's packaging>=24.2.
    export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"

    # Keep Cutlass DSL and quack versioning owned by TokenSpeed's requirements.
    # Duplicating those exact pins here makes the install unsatisfiable when
    # TokenSpeed advances the compatible pair together.

    # Preseed build-time tooling: ``./python`` and ``tokenspeed-kernel`` use
    # ``setuptools.build_meta`` without declaring ``setuptools`` in
    # ``build-system.requires``, and we install with ``--no-build-isolation``.
    uv pip install setuptools wheel pybind11

    # Install the CUDA-13 torch build explicitly (the +cu130 local wheel) before the
    # --no-build-isolation kernel compile below, so the build links matching CUDA 13
    # headers instead of the default PyPI (cu12.x) torch. Pin tracks TokenSpeed's
    # torch requirement; bump alongside the ref in .github/versions/tokenspeed.ref.
    uv pip install "torch==2.11.0+cu130"

    # The kernel's host-stub compile binds crt/host_runtime.h from torch's bundled
    # cu13 headers (site-packages/nvidia/cu*/include/crt) no matter the -I order,
    # and those are a newer patch (nvidia-cuda-runtime 13.0.96) than the apt system
    # nvcc (13.0.88): the 88 nvcc emits a 2-arg __cudaLaunch stub the 96 header's
    # 1-arg macro can't satisfy -> "'__cudaLaunch' was not declared". Those crt dirs
    # are pulled by the kernel build's own dependency resolution, so materialize
    # them with a first build pass (tolerate its compile failure), realign every
    # bundled crt to the system toolkit, then build for real -- deps are satisfied
    # now, so nothing re-pulls the crt.
    uv pip install -e tokenspeed-kernel/python/ --no-build-isolation || \
        echo "first kernel build pass failed (expected: crt skew); realigning crt headers"

    local _sys_crt="${CUDA_HOME}/include/crt"
    local _purelib
    _purelib="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    if [ -d "$_sys_crt" ] && [ -d "$_purelib" ]; then
        local _aligned=0
        while IFS= read -r -d '' _pip_crt; do
            echo "Aligning bundled CUDA crt to system: ${_pip_crt} -> ${_sys_crt}"
            rm -rf "$_pip_crt"
            ln -sfnT "$_sys_crt" "$_pip_crt"
            _aligned=1
        done < <(find "$_purelib" -type d -path '*/nvidia/cu*/include/crt' -print0 2>/dev/null)
        [ "$_aligned" = 1 ] || echo "WARNING: no bundled nvidia crt dirs found under ${_purelib}" >&2
    fi

    uv pip install -e tokenspeed-kernel/python/ --no-build-isolation
    uv pip install -e tokenspeed-scheduler/
    uv pip install -e "./python" --no-build-isolation

    cd "$REPO_ROOT"
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
    uv pip install -e crates/grpc_client/python/
    uv pip install -e grpc_servicer/
}

# ── Main ───────────────────────────────────────────────────────────────────

# Prebuilt fast path: the stamp must match the pinned ref AND the engine must
# actually import in the active venv (a stale image or a venv the baked-venv
# adoption skipped degrades to the source build, never to a broken lane).
use_prebuilt=0
if [ "${TOKENSPEED_BUILD_ONLY:-0}" != "1" ] && [ "${TOKENSPEED_FORCE_SOURCE:-0}" != "1" ] \
    && [ -f "$TOKENSPEED_PREBUILT_STAMP" ] \
    && [ "$(tr -d '[:space:]' < "$TOKENSPEED_PREBUILT_STAMP")" = "$TOKENSPEED_REF" ] \
    && python3 -c "import tokenspeed" &> /dev/null; then
    use_prebuilt=1
    echo "Prebuilt TokenSpeed ${TOKENSPEED_REF} found (stamp ${TOKENSPEED_PREBUILT_STAMP}); skipping source build"
fi

setup_cuda_env

if [ "$use_prebuilt" = "0" ]; then
    install_tokenspeed_from_source
fi

if [ "${TOKENSPEED_BUILD_ONLY:-0}" = "1" ]; then
    # Image build: stamp the built ref for the per-job fast-path check and
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
