#!/bin/bash
# Install protoc from its GitHub release, for a host whose Ubuntu archive does
# not answer (ci_install_build_deps.sh falls back to this when protoc is the
# one tool missing).
#
# The release is the one the archive ships (protobuf 3.21.12 is release
# 21.12), for the host's architecture, checked against a pinned sha256 before
# anything is unpacked, installed under a prefix of the job's own and put on
# GITHUB_PATH for the steps that follow. The caller's own PATH is not changed:
# it prepends <prefix>/bin itself.
#
# Usage: ci_install_protoc.sh
#
# Environment:
#   CI_PROTOC_PREFIX    install prefix (default $HOME/.local).
#   CI_PROTOC_URL_BASE  where the release asset is fetched from (default the
#                       GitHub release; a file:// base in the tests).
#   CI_PROTOC_SHA256    the asset's expected sha256 (default the pinned one
#                       for the host's architecture; the tests pin their own).

set -euo pipefail

VERSION="21.12"
PREFIX="${CI_PROTOC_PREFIX:-${HOME}/.local}"
BASE="${CI_PROTOC_URL_BASE:-https://github.com/protocolbuffers/protobuf/releases/download/v${VERSION}}"

log() { echo "ci_install_protoc: $*"; }

case "$(uname -m)" in
    x86_64)
        arch="x86_64"
        pinned="3a4c1e5f2516c639d3079b1586e703fc7bcfa2136d58bda24d1d54f949c315e8" ;;
    aarch64 | arm64)
        arch="aarch_64"
        pinned="2dd17f75d66a682640b136e31848da9fb2eefe68d55303baf8b32617374f6711" ;;
    *)
        log "no release asset for $(uname -m)" >&2
        exit 1 ;;
esac
expected="${CI_PROTOC_SHA256:-${pinned}}"
asset="protoc-${VERSION}-linux-${arch}.zip"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

curl -fsSL --retry 3 --retry-delay 5 -o "${work}/${asset}" "${BASE}/${asset}"
if ! echo "${expected}  ${work}/${asset}" | sha256sum -c - >/dev/null 2>&1; then
    log "${asset} does not match its pinned sha256; nothing installed" >&2
    exit 1
fi

mkdir -p "${PREFIX}"
unzip -oq "${work}/${asset}" -d "${PREFIX}" 'bin/protoc' 'include/*'
chmod 755 "${PREFIX}/bin/protoc"
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "${PREFIX}/bin" >> "${GITHUB_PATH}"
fi
log "installed $("${PREFIX}/bin/protoc" --version) from ${asset} under ${PREFIX}"
