#!/usr/bin/env bash
# Writes this example as a repository of its own: a controller outside this
# workspace that imports the crate by git, as the guide's line does.
#
#   own-repository.sh <directory> [<git URL of this repository> | --snapshot]
#
# <directory> must not exist. The directory above it must, outside this
# checkout. It gets the example's source, document and tests, a manifest that
# names the crate by git, and this checkout's toolchain pin and lock file.
# The lock keeps every third-party crate at the release this repository
# tests with; cargo rewrites it for the new package on the first build. The
# URL defaults to the project's repository.
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

# The two test-only dependencies, at the versions this workspace names.
serde_yaml="$(sed -n 's/^serde_yaml = "\([^"]*\)"$/\1/p' "${root}/Cargo.toml")"
tokio="$(sed -n 's/^tokio = { version = "\([^"]*\)".*/\1/p' "${root}/Cargo.toml")"
if [ -z "${serde_yaml}" ] || [ -z "${tokio}" ]; then
    echo "own-repository.sh: the workspace manifest names no serde_yaml or tokio version" >&2
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
cat > "${dest}/Cargo.toml" <<TOML
[package]
name = "policy-controller-example"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
smg-policy-controller = { git = "${url}", features = ["controller"] }

[dev-dependencies]
serde_yaml = "${serde_yaml}"
tokio = { version = "${tokio}", features = ["macros", "rt"] }
TOML
echo "own-repository.sh: wrote ${dest} (the crate by git from ${url})"
