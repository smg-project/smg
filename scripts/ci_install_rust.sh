#!/bin/bash
set -euxo pipefail

# The native dependencies (the compiler, pkg-config, the OpenSSL headers,
# protoc): only what the host lacks comes from the archive, and protoc comes
# from its release when the archive does not answer (ci_install_build_deps.sh).
bash "$(dirname "${BASH_SOURCE[0]}")/ci_install_build_deps.sh"
# The release fallback installs under this prefix; the steps that follow read
# it from GITHUB_PATH, this script from PATH.
export PATH="${CI_PROTOC_PREFIX:-${HOME}/.local}/bin:${PATH}"

# Install rustup (Rust installer and version manager). One failed request to
# the release host would end the job, so the install is retried; a later
# attempt resumes the downloads the earlier one kept. The host fails in
# bursts of minutes: seven attempts, with pauses that double from 10 s to
# 320 s (630 s in all), outlast one. An install that works waits for none.
bash "$(dirname "${BASH_SOURCE[0]}")/ci_retry.sh" 7 10 bash -o pipefail -c \
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.100.0"


# Follow the installation prompts, then reload your shell
. "$HOME/.cargo/env"
source $HOME/.cargo/env

# Add cargo bin to GITHUB_PATH so subsequent steps can find rustc/cargo
echo "$HOME/.cargo/bin" >> $GITHUB_PATH

# Verify installation
rustc --version
cargo --version
protoc --version
