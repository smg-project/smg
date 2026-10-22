#!/bin/bash
# Install e2e test dependencies
# Usage: ci_install_e2e_deps.sh [extra_deps...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Installing e2e test dependencies..."
$RETRY 3 10 python3 -m pip install e2e_test/

# Install SmgClient (pure Python client for cross-SDK parity testing)
echo "Installing smg-client..."
$RETRY 3 10 python3 -m pip install clients/python/

# Install any extra dependencies passed as arguments
if [ $# -gt 0 ]; then
    echo "Installing extra dependencies: $@"
    $RETRY 3 10 python3 -m pip --no-cache-dir install --upgrade "$@"
fi

# Match the released generator's runtime floor without replacing the image engine.
# These companions retain a protobuf-6 runtime, including ROCm vLLM's pins.
$RETRY 3 10 python3 -m pip install "grpcio-health-checking==1.78.0" "grpcio-reflection==1.78.0"

echo "E2E test dependencies installed"
