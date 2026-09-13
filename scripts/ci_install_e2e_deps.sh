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

# Pin the grpcio generated-code companions to the protobuf-6 stable line.
#
# Engine setup (sglang's `uv pip install --prerelease=allow` in
# ci_install_sglang.sh) can pull prerelease grpcio wheels such as
# grpcio-health-checking / grpcio-reflection ==1.82.0rc1, whose bundled protobuf
# gencode (7.x) is newer than the protobuf 6.x runtime the engine stack pins.
# Anything that then loads those modules dies with "Detected incompatible
# Protobuf Gencode/Runtime versions ... gencode 7.x runtime 6.x" — e2e workers at
# `import grpc_health.v1.health_pb2`, and the sglang gRPC server at
# `grpc_reflection/v1alpha/reflection.proto`. The 1.81.x line caps protobuf<7 and
# ships the 6.x gencode, so it stays compatible. This runs last to override
# whatever engine/extra-dep installs selected. (grpcio core has no generated
# protobuf, so it is left at the engine-selected version.) Drop the cap once the
# stack moves to a protobuf 7 runtime.
$RETRY 3 10 python3 -m pip install "grpcio-health-checking==1.81.*" "grpcio-reflection==1.81.*"

echo "E2E test dependencies installed"
