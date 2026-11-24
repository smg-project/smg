#!/usr/bin/env bash
# Writes this example as a repository of its own: a controller outside this
# workspace that imports the crate by git, as the guide's line does.
#
#   own-repository.sh <directory> [<git URL of this repository> | --snapshot]
#
# <directory> must not exist. The directory above it must, outside this
# checkout. It gets the example's source, document and tests, a manifest that
# names the crate by git and holds this workspace's release profile, the
# allocator's page setting for Linux aarch64, and this checkout's toolchain
# pin and lock file.
# The lock keeps every third-party crate at the release this repository
# tests with; cargo rewrites it for the new package on the first build. The
# URL defaults to the project's repository.
#
# With a URL the crate is that repository's default branch at its tip: the
# first build resolves the tip and the lock keeps it from then on. The
# copy's source is this checkout's, so run the script from a checkout of
# that tip, or name a `rev` or a `tag` beside `git` in the manifest.
#
# With --snapshot the git source is <directory>-source: a repository of one
# revision, made here from this checkout's tracked files as its index holds
# them. CI builds the copy that way, against the change under test. A hosted
# checkout is a shallow clone, and git fetches from none.
#
# Then, in <directory>: cargo test
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd "${here}/../.." && pwd -P)"
dest="${1:?usage: own-repository.sh <directory> [<git URL of this repository> | --snapshot]}"
url="${2:-https://github.com/smg-project/smg}"

if [ -e "${dest}" ]; then
    echo "own-repository.sh: ${dest} exists" >&2
    exit 2
fi
parent="$(cd "$(dirname "${dest}")" 2>/dev/null && pwd -P)" || {
    echo "own-repository.sh: the directory above ${dest} does not exist" >&2
    exit 2
}
dest="${parent%/}/$(basename "${dest}")"
case "${dest}/" in
    "${root}/"*)
        echo "own-repository.sh: ${dest} is inside this checkout, whose workspace would claim the package" >&2
        exit 2
        ;;
esac

# The allocator and the test-only dependencies, as this workspace names them.
jemallocator="$(sed -n 's/^tikv-jemallocator = \(.*\)$/\1/p' "${root}/Cargo.toml")"
serde_yaml="$(sed -n 's/^serde_yaml = "\([^"]*\)"$/\1/p' "${root}/Cargo.toml")"
tokio="$(sed -n 's/^tokio = { version = "\([^"]*\)".*/\1/p' "${root}/Cargo.toml")"
tonic="$(sed -n 's/^tonic = \(.*\)$/\1/p' "${root}/Cargo.toml")"
if [ -z "${jemallocator}" ] || [ -z "${serde_yaml}" ] || [ -z "${tokio}" ] || [ -z "${tonic}" ]; then
    echo "own-repository.sh: the workspace manifest names no tikv-jemallocator, serde_yaml, tokio or tonic" >&2
    exit 1
fi
# The release profile this workspace builds the stock controller with: its
# keys up to the section's first blank line, their comments dropped.
profile="$(sed -n '/^\[profile\.release\]$/,/^$/{ s/[[:space:]]*#.*$//; /^[a-z-][a-z-]* = /p; }' "${root}/Cargo.toml")"
if [ -z "${profile}" ]; then
    echo "own-repository.sh: the workspace manifest names no release profile" >&2
    exit 1
fi
# The allocator's page setting this workspace builds with: one Linux aarch64
# binary runs on kernels of 4, 16 and 64 KiB pages.
page="$(grep '^AARCH64_UNKNOWN_LINUX_GNU_JEMALLOC_SYS_WITH_LG_PAGE = ' "${root}/.cargo/config.toml" || true)"
if [ -z "${page}" ]; then
    echo "own-repository.sh: the workspace's cargo configuration names no page setting for the allocator" >&2
    exit 1
fi

if [ "${url}" = "--snapshot" ]; then
    snapshot="${dest}-source"
    if [ -e "${snapshot}" ]; then
        echo "own-repository.sh: ${snapshot} exists" >&2
        exit 2
    fi
    git -C "${root}" checkout-index --all --prefix="${snapshot}/"
    git -c init.defaultBranch=main -C "${snapshot}" init --quiet
    git -C "${snapshot}" add --all --force
    # The revision is this script's own: no hook of the host runs on it and
    # a host that signs every revision signs none here.
    git -C "${snapshot}" -c user.name=own-repository -c user.email=own-repository@localhost \
        -c core.hooksPath=/dev/null -c commit.gpgsign=false \
        commit --quiet --message "A snapshot of the checkout"
    url="file://${snapshot}"
fi

mkdir "${dest}"
cp -R "${here}/src" "${here}/tests" "${here}/policy.yaml" "${dest}/"
cp "${root}/Cargo.lock" "${root}/rust-toolchain.toml" "${dest}/"
mkdir "${dest}/.cargo"
printf '[env]\n%s\n' "${page}" > "${dest}/.cargo/config.toml"
# The test-only dependencies below are the example's own list (the manifest
# beside this script), kept here by hand: one added there is added here.
cat > "${dest}/Cargo.toml" <<TOML
[package]
name = "policy-controller-example"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
smg-policy-controller = { git = "${url}", features = ["controller"] }

[target.'cfg(all(not(target_env = "msvc"), not(target_env = "musl"), not(target_vendor = "apple")))'.dependencies]
tikv-jemallocator = ${jemallocator}

[dev-dependencies]
mock-worker = { git = "${url}" }
serde_yaml = "${serde_yaml}"
smg-grpc-client = { git = "${url}" }
smg-policy-controller = { git = "${url}", features = ["client"] }
smg-worker = { git = "${url}", features = ["runtime", "test-util"] }
tokio = { version = "${tokio}", features = ["macros", "net", "rt-multi-thread", "sync", "test-util", "time"] }
tonic = ${tonic}

[profile.release]
${profile}
TOML
echo "own-repository.sh: wrote ${dest} (the crate by git from ${url})"
