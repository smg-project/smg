#!/bin/bash
# Set up an e2e job that runs inside vLLM's ROCm image (the AMD lanes).
#
# The image already ships vLLM with its ROCm stack, NIXL, UCX and MoRI, so this
# builds a venv on top of the image's packages instead of installing an engine.
# Job-specific packages live in a venv; the engine keeps its image version.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

if [[ -n "${SMG_CI_RUNTIME_STATE_DIR:-}" ]]; then
    SMG_CI_EFFECTIVE_CONSTRAINT="$(python3 "$SCRIPT_DIR/ci_runtime_constraints.py" prepare "$SMG_CI_RUNTIME_STATE_DIR")"
    export PIP_CONSTRAINT="$SMG_CI_EFFECTIVE_CONSTRAINT"
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        printf '%s\n' "PIP_CONSTRAINT=$PIP_CONSTRAINT" >> "$GITHUB_ENV"
    fi
fi

SMG_CI_IMAGE_PACKAGE_VERSIONS="$(python3 -c '
from importlib.metadata import version
from packaging.version import Version
print(Version(version("vllm")), Version(version("grpcio-reflection")))
')"
read -r VLLM_VERSION SMG_CI_GRPC_COMPANION_VERSION <<< "$SMG_CI_IMAGE_PACKAGE_VERSIONS"

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
echo "image: $(python3 -V), ROCm $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"

# The image's PyGObject package needs Pycairo and its native Cairo library.
if ! python3 -c '
import cairo
from importlib.metadata import version
from packaging.version import Version
assert Version(version("pycairo")) >= Version("1.16.0")
' >/dev/null 2>&1; then
    if ! command -v pkg-config >/dev/null || ! pkg-config --exists 'cairo >= 1.15.10'; then
        if [[ "${GITHUB_ACTIONS:-}" != true || "$(id -u)" != 0 || ! -f /.dockerenv ]]; then
            echo "ERROR: Cairo prerequisites require the root Actions container" >&2
            exit 1
        fi
        $RETRY 3 10 apt-get update
        $RETRY 3 10 apt-get install -y --no-install-recommends pkg-config libcairo2-dev
    fi
    SMG_CI_PYCAIRO_SOURCE='https://files.pythonhosted.org/packages/bb/5e/19ab2572981a996ce96b853a189d2dcced40506b67d9555a7568457abcf2/pycairo-1.29.2.tar.gz#sha256=3e69fff74fe64f5ba2dfa31f67c6bdf26413342574047437d2ac520d35e9a489'
    $RETRY 3 10 python3 -m pip install --retries 0 --no-deps --no-cache-dir --no-binary=pycairo "$SMG_CI_PYCAIRO_SOURCE"
    python3 -c 'import cairo; print(cairo.cairo_version_string())'
fi

# Install gRPC packages from source (not PyPI) so PR changes are always tested
$RETRY 3 10 pip install "vllm==${VLLM_VERSION}" -e crates/grpc_client/python/ -e grpc_servicer/

# The image ships pytest too; without a copy in the venv, `pytest` is the image's
# script on the system interpreter, which cannot see the venv's packages.
$RETRY 3 10 pip install --ignore-installed pytest

python3 -c "import torch, vllm, smg_grpc_servicer; assert torch.version.hip; print('vllm', vllm.__version__)"

if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$PWD/.venv/bin" >> "$GITHUB_PATH"
fi

# Carry this image's companion version into later steps of this Actions job.
# Standalone callers must pass the same override to ci_install_e2e_deps.sh.
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SMG_CI_GRPC_COMPANION_VERSION=${SMG_CI_GRPC_COMPANION_VERSION}" >> "$GITHUB_ENV"
fi
