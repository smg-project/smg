#!/bin/bash
set -euxo pipefail

# The native dependencies (the compiler, pkg-config, the OpenSSL headers,
# protoc): only what the host lacks comes from the archive, and protoc comes
# from its release when the archive does not answer (ci_install_build_deps.sh).
bash "$(dirname "${BASH_SOURCE[0]}")/ci_install_build_deps.sh"
# The release fallback installs under this prefix; the steps that follow read
# it from GITHUB_PATH, this script from PATH.
export PATH="${CI_PROTOC_PREFIX:-${HOME}/.local}/bin:${PATH}"

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
