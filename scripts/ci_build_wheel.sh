#!/bin/bash
# Build Python wheel and Go FFI library in parallel
# This script is used by CI to build both artifacts concurrently.

set -euo pipefail

# Setup Rust environment
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi
export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install maturin and zig for manylinux cross-compilation
python3 -m pip install --upgrade pip maturin ziglang

# Start Go FFI build in background
echo "Starting Go FFI build in background..."
(cd bindings/golang && make build && echo "Go FFI: OK" && ls -la target/release/libsmg_go.*) &
GO_PID=$!

# Build Python wheel in foreground
echo "Building Python wheel..."
cd bindings/python
maturin build --profile ci --locked --features vendored-openssl --manylinux 2_28 --zig --out dist
echo "Python wheel: OK"
ls -lh dist/

# Wait for Go build to complete
echo "Waiting for Go FFI build..."
wait $GO_PID
GO_EXIT=$?
if [ $GO_EXIT -ne 0 ]; then
    echo "Go FFI build failed with exit code $GO_EXIT"
    exit $GO_EXIT
fi

echo "Both builds completed successfully"
