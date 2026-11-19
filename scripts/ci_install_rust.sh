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
# attempt resumes the downloads the earlier one kept. Five attempts with
# waits of 10, 20, 40 and 80 seconds: the host fails by spells of minutes,
# and three attempts with waits of 10 and 20 seconds still lost jobs.
#
# The two curl calls of an attempt, the one below and the installer script's
# own, take no connect limit: a connect that failed held an attempt about 130
# seconds, and the script's curl tries four times. The config file gives both
# the 10 seconds the other fetches of CI ask.
curl_home="${RUNNER_TEMP:-/tmp}/ci-install-rust-curl"
mkdir -p "${curl_home}"
echo "connect-timeout = 10" > "${curl_home}/.curlrc"

# Each attempt is ended after 180 seconds, so one that never answers is tried
# again instead of holding its step. Five attempts and their waits then end
# within 1,050 seconds, inside the limit of every job that runs the install.
CURL_HOME="${curl_home}" bash "$(dirname "${BASH_SOURCE[0]}")/ci_retry.sh" 5 10 timeout 180 bash -o pipefail -c \
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
