#!/bin/bash
# Setup Python venv for CI jobs
# Creates a virtual environment on a PINNED interpreter and adds it to GITHUB_PATH

set -euo pipefail

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

HOST_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "none")"

if [ "$HOST_VERSION" = "$PY_VERSION" ]; then
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
        sudo apt-get update
        sudo apt-get install -y python3-pip python3-venv
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
        echo "Installing uv $UV_VERSION..."
        curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv python install "$PY_VERSION"
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
    echo "CUDA_HOME=/usr/local/cuda" >> "$GITHUB_ENV"
else
    echo "Activate venv with: source .venv/bin/activate"
fi

echo "Python venv setup complete"
