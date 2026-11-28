#!/bin/bash
# Build the example policy plugin: a core module for wasm32-unknown-unknown,
# wrapped as a component of the smg:policy record world.
#
# Not wasm32-wasip2, the target of the other guests in this directory: the
# policy host has no WASI and refuses a component that imports any `wasi:`
# interface, which a wasip2 build of std carries.

set -e

cd "$(dirname "$0")"

command -v cargo >/dev/null 2>&1 || { echo "Error: cargo is required but not installed." >&2; exit 1; }

# With rustup the target is added when it is missing. A toolchain that did
# not come from rustup is left alone: cargo names a missing target itself.
if command -v rustup >/dev/null 2>&1; then
    if ! rustup target list --installed | grep -q "^wasm32-unknown-unknown$"; then
        echo "Installing the wasm32-unknown-unknown target..."
        rustup target add wasm32-unknown-unknown
    fi
fi

if ! command -v wasm-tools >/dev/null 2>&1; then
    echo "Error: wasm-tools is required but not installed." >&2
    echo "Install it with: cargo install wasm-tools" >&2
    exit 1
fi

# The lock file is committed: the example builds the same tomorrow.
cargo build --release --locked --target wasm32-unknown-unknown --target-dir target

MODULE="target/wasm32-unknown-unknown/release/wasm_guest_policy.wasm"
COMPONENT="target/wasm32-unknown-unknown/release/wasm_guest_policy.component.wasm"

# The module carries the world it was built against; this wraps it as the
# component the host loads.
wasm-tools component new "$MODULE" -o "$COMPONENT"

echo "Built $(pwd)/$COMPONENT ($(wc -c < "$COMPONENT") bytes)"
