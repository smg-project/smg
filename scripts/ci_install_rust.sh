#!/bin/bash
set -euxo pipefail

# Fix any half-configured packages (e.g. cuda-keyring left from previous runs)
export DEBIAN_FRONTEND=noninteractive
if command -v sudo >/dev/null 2>&1; then
    sudo dpkg --configure -a --force-confnew 2>/dev/null || true
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev pkg-config protobuf-compiler
else
    dpkg --configure -a --force-confnew 2>/dev/null || true
    apt-get update
    apt-get install -y build-essential libssl-dev pkg-config protobuf-compiler
fi

# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.98.0


# Follow the installation prompts, then reload your shell
. "$HOME/.cargo/env"
source $HOME/.cargo/env

# Add cargo bin to GITHUB_PATH so subsequent steps can find rustc/cargo
echo "$HOME/.cargo/bin" >> $GITHUB_PATH

# Verify installation
rustc --version
cargo --version
protoc --version
