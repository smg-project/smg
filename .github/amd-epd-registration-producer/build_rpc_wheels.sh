#!/bin/bash
set -euo pipefail

# Build the unpublished packages from this checkout, separately from the Rust wheel.
PACKAGE_SOURCE="$RUNNER_TEMP/epd-rpc-source"
mkdir "$PACKAGE_SOURCE"
git -c "safe.directory=$GITHUB_WORKSPACE" archive HEAD \
    crates/grpc_client/proto crates/grpc_client/python grpc_servicer \
    | tar -xf - -C "$PACKAGE_SOURCE"
python3 -m venv "$RUNNER_TEMP/epd-rpc-build"
PYTHON="$RUNNER_TEMP/epd-rpc-build/bin/python"
"$PYTHON" -m pip install setuptools==81.0.0 grpcio-tools==1.78.0
mkdir epd-rpc-wheels
"$PYTHON" -m pip wheel --no-deps --no-build-isolation --wheel-dir epd-rpc-wheels \
    "$PACKAGE_SOURCE/crates/grpc_client/python" "$PACKAGE_SOURCE/grpc_servicer"
