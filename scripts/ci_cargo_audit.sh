#!/bin/bash
# Audit Cargo.lock against the RustSec advisory database with a pinned cargo-audit release.
#
# Used by the cargo-audit job of pr-test-rust.yml and by cargo-audit-nightly.yml; run it locally
# from the repository root the same way: `bash scripts/ci_cargo_audit.sh [lockfile] [cargo-audit args]`.
# The binary is a release archive from https://github.com/rustsec/rustsec/releases, verified against
# the checksums below (no compile, no third-party action). The ignore list is .cargo/audit.toml; an
# entry there needs a reason. The exit status is cargo-audit's: non-zero on any vulnerability that
# is not ignored, zero when only warnings (unmaintained, yanked) remain.
set -euo pipefail

CARGO_AUDIT_VERSION=0.22.2
case "$(uname -m)" in
  x86_64)  target=x86_64-unknown-linux-musl; sha256=7fb9497f8594b389e5fce5ef9b92db08432996895b2e0c5a0167a69ed445c428 ;;
  aarch64) target=aarch64-unknown-linux-gnu; sha256=c6603814ddaa45e51263dafd31c0ac98808f688d26f7395804f9670b0fd599dd ;;
  *) echo "ci_cargo_audit.sh: no pinned cargo-audit release for $(uname -m)" >&2; exit 2 ;;
esac

name="cargo-audit-${target}-v${CARGO_AUDIT_VERSION}"
# The archive is kept in a per-user directory (never a shared /tmp), checked against the pinned
# checksum and extracted afresh on every run: a binary left there by anyone or by an interrupted
# extraction is never run as it is.
bin_dir="${CARGO_AUDIT_BIN_DIR:-${RUNNER_TEMP:-${XDG_CACHE_HOME:-$HOME/.cache}}/cargo-audit-${CARGO_AUDIT_VERSION}}"
bin="${bin_dir}/cargo-audit"
archive="${bin_dir}/${name}.tgz"
mkdir -p "${bin_dir}"
# Downloaded whenever the cached archive is missing or does not match the pinned checksum (a
# download that died partway, a tampered file): a bad archive is replaced, never kept.
if ! echo "${sha256}  ${archive}" | sha256sum --check --quiet 2>/dev/null; then
  rm -f "${archive}"
  curl -fsSL --retry 3 --connect-timeout 10 --max-time 300 -o "${archive}" \
    "https://github.com/rustsec/rustsec/releases/download/cargo-audit%2Fv${CARGO_AUDIT_VERSION}/${name}.tgz"
  echo "${sha256}  ${archive}" | sha256sum --check --quiet
fi
rm -f "${bin}"
tar xzf "${archive}" --strip-components=1 -C "${bin_dir}" "${name}/cargo-audit"

"${bin}" --version
lockfile="${1:-Cargo.lock}"
shift $(( $# > 0 ? 1 : 0 ))
exec "${bin}" audit --file "${lockfile}" "$@"
