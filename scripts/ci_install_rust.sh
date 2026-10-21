#!/bin/bash
set -euxo pipefail

# Fix any half-configured packages (e.g. cuda-keyring left from previous runs)
export DEBIAN_FRONTEND=noninteractive
# The index refresh survives a third-party source of the runner image that
# does not answer (scripts/ci_apt_update.sh): the packages below come from the
# Ubuntu archive alone.
if command -v sudo >/dev/null 2>&1; then
    sudo dpkg --configure -a --force-confnew 2>/dev/null || true
    bash "$(dirname "${BASH_SOURCE[0]}")/ci_apt_mirror.sh"
    bash "$(dirname "${BASH_SOURCE[0]}")/ci_apt_update.sh" sudo
    sudo apt-get install -y build-essential libssl-dev pkg-config protobuf-compiler
else
    dpkg --configure -a --force-confnew 2>/dev/null || true
    bash "$(dirname "${BASH_SOURCE[0]}")/ci_apt_mirror.sh"
    bash "$(dirname "${BASH_SOURCE[0]}")/ci_apt_update.sh"
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
