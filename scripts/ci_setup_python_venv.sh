#!/bin/bash
# Setup Python venv for CI jobs
# Creates a virtual environment on a PINNED interpreter and adds it to GITHUB_PATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

# Every CI lane must agree on the interpreter. A bare `python3 -m venv` resolves
# to whatever the host ships (3.12 in the containerised pools, 3.10 on the
# bare-metal GPU runners), so which Python a job ran on depended on which
# machine it landed on -- and a 3.10-only breakage can take out the nightly
# while regular CI, which never sees 3.10, stays green.
#
# Override only for a deliberate older-interpreter lane, never to work around a
# host missing the pinned version -- that reintroduces the drift.
PY_VERSION="${CI_PYTHON_VERSION:-3.12}"

# Pinned so the job's toolchain does not change under it between runs.
UV_VERSION="${UV_VERSION:-0.12.5}"

install_uv() {
    echo "Installing uv $UV_VERSION..."
    $RETRY 3 5 bash -c "set -o pipefail; curl -LsSf 'https://astral.sh/uv/${UV_VERSION}/install.sh' | sh"
    export PATH="$HOME/.local/bin:$PATH"
}

# CPU image builds need uv's driver-profile override even when host Python
# already matches. Reuse this script's pin instead of the floating installer.
if [ "${SMG_BUILD_PREPARED_ENV:-0}" = 1 ]; then
    install_uv
fi

# Image builds run as root, even when the base image also provides sudo.
if [ "$(id -u)" != 0 ] && command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi

HOST_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "none")"

# vLLM and SGLang runner images have separate, Pod-local environments. Only
# adopt one when its installation recipe still matches this checkout; stale
# images fall back to the existing fresh-venv installation path.
if [ -n "${SMG_CI_BACKEND:-}" ]; then
    source "${SCRIPT_DIR}/ci_prepared_backend_env.sh"
    case "$SMG_CI_BACKEND" in vllm|sglang) ;; *) echo "Unknown backend: $SMG_CI_BACKEND" >&2; exit 1 ;; esac
    PREPARED_VENV="${SMG_PREINSTALLED_ROOT:-/opt/smg-ci}/${SMG_CI_BACKEND}/.venv"
    SMG_BAKED_VENV=""
    if smg_prepared_env_matches "$SMG_CI_BACKEND" "$PREPARED_VENV"; then
        SMG_BAKED_VENV="$PREPARED_VENV"
    else
        echo "Prepared $SMG_CI_BACKEND environment unavailable or incompatible; using a fresh environment"
    fi
fi

# Prebuilt CI images (docker/ci-tokenspeed.Dockerfile) bake a fully-provisioned
# venv and advertise it via SMG_BAKED_VENV. Adopt it as ./.venv so every
# downstream step (the GITHUB_PATH entry below, pip installs, pytest) uses the
# baked interpreter + packages transparently. Adoption is conditional on the
# interpreter matching the pin: a stale image falls through to a fresh venv
# here, and scripts/ci_install_tokenspeed.sh installs nightly wheels.
ADOPTED_VENV=""
if [ -n "${SMG_BAKED_VENV:-}" ] && [ -x "${SMG_BAKED_VENV}/bin/python" ]; then
    BAKED_VERSION="$("${SMG_BAKED_VENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$BAKED_VERSION" = "$PY_VERSION" ]; then
        echo "Adopting baked venv ${SMG_BAKED_VENV} (python ${BAKED_VERSION})"
        # No trailing slash: removes a leftover .venv dir or symlink, never
        # the baked venv's contents.
        rm -rf .venv
        ln -s "${SMG_BAKED_VENV}" .venv
        ADOPTED_VENV=1
    else
        echo "WARNING: baked venv is python ${BAKED_VERSION}, pin is ${PY_VERSION} — ignoring it" >&2
    fi
fi

if [ -n "$ADOPTED_VENV" ]; then
    : # venv adopted above; the assertions below still validate it.
elif [ "$HOST_VERSION" = "$PY_VERSION" ]; then
    # Use the host interpreter directly: a no-op for every lane that already
    # ships the pinned version -- nothing installed, nothing downloaded.
    echo "Host python3 is $PY_VERSION - creating venv with it"
    # Attempt creation and repair only on real failure. Pre-flight checks get
    # this wrong: `python3 -m venv --help` succeeds on Debian hosts where
    # creation will fail, because the venv module and ensurepip ship in
    # separate packages.
    if ! python3 -m venv .venv; then
        if ! command -v apt-get &> /dev/null; then
            echo "ERROR: cannot create a venv, and apt-get is not available to repair it" >&2
            exit 1
        fi
        echo "venv creation failed - installing python3-venv/python3-pip, then retrying"
        bash "${SCRIPT_DIR}/ci_apt_mirror.sh"
        $RETRY 3 10 $SUDO apt-get update
        $RETRY 3 10 $SUDO apt-get install -y python3-pip python3-venv
        rm -rf .venv
        python3 -m venv .venv
    fi
else
    # Provision the pinned interpreter with uv instead of mutating the machine:
    # a standalone CPython in the user cache, no sudo, system python untouched.
    echo "Host python3 is $HOST_VERSION - provisioning $PY_VERSION with uv"
    if ! command -v uv &> /dev/null; then
        # Version-pinned installer URL. This script runs before the others that
        # install uv and they all skip when it is present, so the pin holds for
        # the whole job.
        install_uv
    fi
    $RETRY 3 10 uv python install "$PY_VERSION"
    # --seed: a uv venv ships without pip, and downstream CI steps run
    # `python3 -m pip install` inside this venv.
    uv venv --python "$PY_VERSION" --seed .venv
fi

# Assert the invariant instead of trusting it: a venv on the wrong interpreter
# must fail here, not as an unrelated-looking import error in a later step.
ACTUAL_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$ACTUAL_VERSION" != "$PY_VERSION" ]; then
    echo "ERROR: venv interpreter is $ACTUAL_VERSION, expected $PY_VERSION" >&2
    exit 1
fi
if ! .venv/bin/python -m pip --version &> /dev/null; then
    echo "ERROR: venv has no pip; downstream steps call 'python3 -m pip install'" >&2
    exit 1
fi

echo "venv interpreter: $ACTUAL_VERSION (pinned)"

# Add to GitHub Actions PATH if running in CI
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$PWD/.venv/bin" >> "$GITHUB_PATH"
    CI_CUDA_HOME=/usr/local/cuda
    if [ -n "$ADOPTED_VENV" ] && [ "${SMG_CI_BACKEND:-}" = sglang ]; then
        CI_CUDA_HOME=/usr/local/cuda-13.0
    fi
    echo "CUDA_HOME=$CI_CUDA_HOME" >> "$GITHUB_ENV"
    # Expose the host CUDA toolkit when there is one. vLLM only enables its
    # FlashInfer paths when `nvcc` is on PATH (or flashinfer-cubin is installed,
    # which no longer ships for current FlashInfer releases); the pip
    # nvidia-cuda-nvcc wheel lands under site-packages, where shutil.which never
    # looks. With the toolkit installed but off PATH, a bare-metal runner runs
    # every lane with FlashInfer silently disabled, and the pinned vLLM's MXFP8
    # kernel selector then picks a FlashInfer kernel it cannot run
    # ("module 'vllm.utils.flashinfer' has no attribute 'mm_mxfp8'"). The k8s
    # GPU images ship the CUDA runtime but no toolkit, so this is a no-op there.
    if [ -x "$CI_CUDA_HOME/bin/nvcc" ]; then
        echo "$CI_CUDA_HOME/bin" >> "$GITHUB_PATH"
    fi
else
    echo "Activate venv with: source .venv/bin/activate"
fi

echo "Python venv setup complete"
