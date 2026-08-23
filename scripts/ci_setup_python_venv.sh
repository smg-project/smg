#!/bin/bash
# Setup Python venv for CI jobs
# Creates a virtual environment on a PINNED interpreter and adds it to GITHUB_PATH

set -euo pipefail

# The interpreter every CI lane must agree on.
#
# This used to be a bare `python3 -m venv`, which resolves to whatever the host
# image happens to ship: 3.12 in the containerised pools, 3.10 on the
# bare-metal GPU runners. Which Python a job ran on therefore depended on which
# machine it landed on, and nothing declared or checked that.
#
# The cost of that is not theoretical. A pinned upstream package that is
# unimportable on 3.10 took out every bare-metal leg of the tau2 nightly for
# five consecutive runs while regular CI stayed green -- because regular CI
# never runs 3.10, so it could not have caught it. Any nightly-only failure of
# that shape is indistinguishable from a nightly bug until someone compares
# interpreters.
#
# Override for a deliberate older-interpreter lane. Do NOT override it to work
# around a machine that is missing the pinned version -- that reintroduces the
# drift this exists to prevent.
PY_VERSION="${CI_PYTHON_VERSION:-3.12}"

host_python_version() {
    python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "none"
}

HOST_VERSION="$(host_python_version)"

if [ "$HOST_VERSION" = "$PY_VERSION" ]; then
    # The host already ships the pinned version, so use it directly. This is
    # the path every currently-green lane takes: pinning is a no-op for them,
    # adds no dependency, and downloads nothing. Only the outlier hosts take
    # the branch below.
    echo "Host python3 is $PY_VERSION - creating venv with it"
    if ! command -v pip3 &> /dev/null || ! python3 -m venv --help &> /dev/null; then
        echo "Installing pip and venv..."
        sudo apt update
        sudo apt install -y python3-pip python3-venv
    fi
    python3 -m venv .venv
else
    # The host ships something else. Provision the pinned interpreter with uv
    # instead of mutating the machine: uv fetches a standalone CPython into the
    # user cache, needs no sudo, and leaves the system python untouched. That
    # keeps this fixable from the repo on hosts whose distro python is too old
    # to upgrade in place.
    echo "Host python3 is $HOST_VERSION - provisioning $PY_VERSION with uv"
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv python install "$PY_VERSION"
    # --seed matters: a uv venv ships without pip, and downstream CI steps call
    # `python3 -m pip install` inside this venv (ci_install_e2e_deps.sh) and
    # `pip install wheel/*.whl` from the workflows. Without it they fail with a
    # missing-module error that looks nothing like its cause.
    uv venv --python "$PY_VERSION" --seed .venv
fi

# Assert the invariant rather than trusting it. Both branches above can produce
# a venv on the wrong interpreter if the environment surprises us, and that is
# precisely the failure being fixed here -- silent, and surfacing much later as
# an unrelated-looking import error in a different script.
ACTUAL_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$ACTUAL_VERSION" != "$PY_VERSION" ]; then
    echo "ERROR: venv interpreter is $ACTUAL_VERSION, expected $PY_VERSION" >&2
    exit 1
fi

# Same reasoning for pip: assert it is present here rather than letting a later
# script be the one to discover it is missing.
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
