#!/bin/bash
# Install the wasm-tools CLI from its GitHub release, for the job that builds
# the example policy plugin: examples/wasm/wasm-guest-policy/build.sh wraps
# the guest's core module as a component with `wasm-tools component new`.
#
# The release is the one of the wit-component crate the policy crate's tests
# depend on (crates/policy_controller/Cargo.toml: 0.x.y there is 1.x.y here),
# for the host's architecture, checked against a pinned sha256 before anything
# is unpacked, installed under a prefix of the job's own and put on
# GITHUB_PATH for the steps that follow. The caller's own PATH is not changed:
# it prepends <prefix>/bin itself.
#
# Usage: ci_install_wasm_tools.sh
#
# Environment:
#   CI_WASM_TOOLS_PREFIX    install prefix (default $HOME/.local).
#   CI_WASM_TOOLS_URL_BASE  where the release asset is fetched from (default
#                           the GitHub release; a file:// base in the tests).
#   CI_WASM_TOOLS_SHA256    the asset's expected sha256 (default the pinned one
#                           for the host's architecture; the tests pin their own).

set -euo pipefail

VERSION="1.258.3"
PREFIX="${CI_WASM_TOOLS_PREFIX:-${HOME}/.local}"
BASE="${CI_WASM_TOOLS_URL_BASE:-https://github.com/bytecodealliance/wasm-tools/releases/download/v${VERSION}}"

log() { echo "ci_install_wasm_tools: $*"; }

case "$(uname -m)" in
    x86_64)
        arch="x86_64"
        pinned="99b486a9ddc19af0260cd5f98f3b709a60817a3acd20efc9c582d719caebbf6d" ;;
    aarch64 | arm64)
        arch="aarch64"
        pinned="4b9d10d460343bf7e13dcfa8ddc1e06661ff1894d4e3caf3920780c5b21dc1dd" ;;
    *)
        log "no release asset for $(uname -m)" >&2
        exit 1 ;;
esac
expected="${CI_WASM_TOOLS_SHA256:-${pinned}}"
name="wasm-tools-${VERSION}-${arch}-linux"
asset="${name}.tar.gz"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

curl -fsSL --retry 3 --retry-delay 5 -o "${work}/${asset}" "${BASE}/${asset}"
if ! echo "${expected}  ${work}/${asset}" | sha256sum -c - >/dev/null 2>&1; then
    log "${asset} does not match its pinned sha256; nothing installed" >&2
    exit 1
fi

tar -xzf "${work}/${asset}" -C "${work}" "${name}/wasm-tools"
mkdir -p "${PREFIX}/bin"
cp "${work}/${name}/wasm-tools" "${PREFIX}/bin/wasm-tools"
chmod 755 "${PREFIX}/bin/wasm-tools"
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "${PREFIX}/bin" >> "${GITHUB_PATH}"
fi
log "installed $("${PREFIX}/bin/wasm-tools" --version) from ${asset} under ${PREFIX}"
