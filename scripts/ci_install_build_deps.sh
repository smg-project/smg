#!/bin/bash
# Install the build's native dependencies, and only the ones the host lacks.
#
# The workspace needs C and C++ compilers and make (the cc crate's builds,
# the tokenizers crate's C++ sources), pkg-config with the OpenSSL headers
# (openssl-sys) and protoc (prost-build in the gRPC crates). The GitHub-hosted
# runner images carry the first three and not protoc, so a job that installed
# all four through apt refreshed the archive's indexes for one package every
# time, and an archive outage (every Ubuntu mirror unresolvable for hours on
# 2026-11-11) held every check of the day for tools the image already had.
# Other runners (the cluster's CPU runners) carry none of the four; there the
# archive installs all of them, as before.
#
# Usage: ci_install_build_deps.sh
#
# What is present is left alone. What is missing comes from the Ubuntu archive
# through the mirror, index-refresh and retry helpers beside this script. When
# protoc alone is missing and the archive does not answer, protoc comes from
# its GitHub release instead (ci_install_protoc.sh, pinned by sha256), which
# appends its bin directory to GITHUB_PATH for the steps that follow; the
# caller puts that directory on its own PATH (ci_install_rust.sh does), as an
# export here would die with this shell.
#
# Environment:
#   CI_APT_INSTALL_RETRY_DELAY  first delay of the install retry, seconds
#                               (default 10).
#   CI_PROTOC_PREFIX            where the release fallback installs protoc
#                               (default $HOME/.local).

set -euo pipefail

here="$(dirname "${BASH_SOURCE[0]}")"
export DEBIAN_FRONTEND=noninteractive

log() { echo "ci_install_build_deps: $*"; }

missing=()
if ! command -v cc >/dev/null 2>&1 || ! command -v c++ >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    missing+=(build-essential)
fi
if ! command -v pkg-config >/dev/null 2>&1; then
    missing+=(pkg-config)
fi
if ! pkg-config --exists openssl 2>/dev/null; then
    missing+=(libssl-dev)
fi
if ! command -v protoc >/dev/null 2>&1; then
    missing+=(protobuf-compiler)
fi

if [ "${#missing[@]}" -eq 0 ]; then
    log "the host has the C and C++ compilers, make, pkg-config, the OpenSSL headers and protoc; apt is not needed"
    exit 0
fi
log "missing: ${missing[*]}"

SUDO=""
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi
# Fix any half-configured packages (a cuda-keyring left from an earlier run).
# shellcheck disable=SC2086 # an empty SUDO must vanish, not become an argument
${SUDO} dpkg --configure -a --force-confnew 2>/dev/null || true

# The index refresh survives a third-party source of the runner image that
# does not answer (ci_apt_update.sh): the packages come from the Ubuntu
# archive alone.
installed=0
bash "${here}/ci_apt_mirror.sh" || log "the mirror probe failed; the sources stay as they are"
# shellcheck disable=SC2086
if bash "${here}/ci_apt_update.sh" ${SUDO} \
    && bash "${here}/ci_retry.sh" 3 "${CI_APT_INSTALL_RETRY_DELAY:-10}" ${SUDO} apt-get install -y "${missing[@]}"; then
    installed=1
fi

if [ "${installed}" -eq 1 ]; then
    log "installed ${missing[*]} from the archive"
elif [ "${#missing[@]}" -eq 1 ] && [ "${missing[0]}" = "protobuf-compiler" ]; then
    log "the archive did not answer; protoc comes from its release"
    bash "${here}/ci_install_protoc.sh"
else
    log "could not install ${missing[*]} from the archive" >&2
    exit 1
fi
