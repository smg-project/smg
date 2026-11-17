#!/bin/bash
# Build WASM test fixtures from the example guest sources.
#
# Run from the repository root:
#   ./wasm/tests/fixtures/build_fixtures.sh
#
# Prerequisites:
#   rustup target add wasm32-wasip2

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
FIXTURES_DIR="$REPO_ROOT/crates/wasm/tests/fixtures"

echo "Building WASM test fixtures..."

# Verify toolchain
command -v cargo >/dev/null 2>&1 || { echo "Error: cargo not found"; exit 1; }
if ! rustup target list --installed | grep -q "wasm32-wasip2"; then
    echo "Installing wasm32-wasip2 target..."
    # One failed download would end the step; a later attempt resumes it.
    # No ceiling on an attempt here, as the CI calls have: this script is run
    # by hand too, where `timeout` may not exist.
    bash "$REPO_ROOT/scripts/ci_retry.sh" 5 10 rustup target add wasm32-wasip2
fi

build_guest() {
    local name="$1"
    local crate_name="$2"
    local src_dir="$REPO_ROOT/examples/wasm/$name"
    local output_name="$3"

    echo "  Building $name..."
    (cd "$src_dir" && cargo build --target wasm32-wasip2 --release --quiet)

    # wit-bindgen 0.21+ produces components directly; find the output
    # in the guest's own target dir (each example has [workspace] so
    # cargo writes to examples/wasm/<name>/target/, not the repo root).
    local wasm="$src_dir/target/wasm32-wasip2/release/${crate_name}.wasm"
    if [ ! -f "$wasm" ]; then
        echo "Error: $wasm not found"
        exit 1
    fi

    cp "$wasm" "$FIXTURES_DIR/$output_name"
    echo "  -> $FIXTURES_DIR/$output_name ($(du -h "$FIXTURES_DIR/$output_name" | cut -f1 | xargs))"
}

build_guest "wasm-guest-storage-hook" \
            "wasm_guest_storage_hook" \
            "storage_hook_guest.wasm"

build_guest "wasm-guest-storage-hook-passthrough" \
            "wasm_guest_storage_hook_passthrough" \
            "storage_hook_passthrough.wasm"

echo "Done. Fixtures ready in $FIXTURES_DIR"
