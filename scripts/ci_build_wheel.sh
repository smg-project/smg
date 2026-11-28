#!/bin/bash
# Build Python wheel for CI.

set -euo pipefail

# Setup Rust environment
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

# Install maturin and zig for manylinux cross-compilation
python3 -m pip install --upgrade pip maturin ziglang

# Build Python wheel
echo "Building Python wheel..."
cd bindings/python
maturin build --profile ci --features vendored-openssl --manylinux 2_28 --zig --out dist
echo "Python wheel: OK"
ls -lh dist/
