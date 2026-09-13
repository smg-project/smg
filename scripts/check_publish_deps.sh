#!/usr/bin/env bash
# Fail before `cargo publish` when a workspace dependency of CRATE is not on
# crates.io at the version the packaged manifest will require.
#
# `cargo publish` rewrites every path dependency into a plain version
# requirement, so a workspace crate that was never published (or not yet
# published at the pinned version) makes the verify build fail with an
# unhelpful "unresolved import" or "no matching package" deep in rustc
# output, and it does so only after the earlier tiers have already run.
# This check names the missing crates up front.
#
# Usage: scripts/check_publish_deps.sh <crate-name>
# Env:   CRATES_INDEX_URL  sparse-index base (default https://index.crates.io),
#                          overridable for tests.
set -euo pipefail

crate="${1:?usage: $0 <crate-name>}"
index="${CRATES_INDEX_URL:-https://index.crates.io}"

# Sparse-index path for a crate name (https://doc.rust-lang.org/cargo/reference/registry-index.html).
index_path() {
    local name="$1"
    case ${#name} in
        1) echo "1/${name}" ;;
        2) echo "2/${name}" ;;
        3) echo "3/${name:0:1}/${name}" ;;
        *) echo "${name:0:2}/${name:2:2}/${name}" ;;
    esac
}

stderr=$(mktemp)
trap 'rm -f "$stderr"' EXIT
if ! metadata=$(cargo metadata --format-version 1 --no-deps --locked 2> "$stderr"); then
    cat "$stderr" >&2
    exit 2
fi
if ! jq -e --arg c "$crate" '.packages[] | select(.name == $c)' <<< "$metadata" > /dev/null; then
    echo "check_publish_deps: no workspace package named '${crate}'" >&2
    exit 2
fi

# Every workspace path dependency that survives packaging: "<name> <req>".
# Dev-dependencies are dropped from the published manifest, so they are
# not checked; normal and build dependencies are.
deps=$(jq -r --arg c "$crate" \
    '.packages[] | select(.name == $c) | .dependencies[] | select(.path != null and .kind != "dev") | "\(.name) \(.req)"' \
    <<< "$metadata" | sort -u)

if [[ -z "$deps" ]]; then
    echo "${crate}: no workspace dependencies to check"
    exit 0
fi

missing=()
while read -r name req; do
    [[ -z "$name" ]] && continue
    version="${req#^}"      # workspace pins are exact: ^X.Y.Z
    body=$(curl -sf "${index}/$(index_path "$name")" || true)
    if [[ -n "$body" ]] && jq -se --arg v "$version" 'any(.[]; .vers == $v and (.yanked | not))' <<< "$body" > /dev/null; then
        echo "  ok       ${name} ${version}"
    else
        echo "  MISSING  ${name} ${version}"
        missing+=("${name} ${version}")
    fi
done <<< "$deps"

if (( ${#missing[@]} )); then
    echo
    echo "${crate}: ${#missing[@]} workspace dependenc$([[ ${#missing[@]} -eq 1 ]] && echo y || echo ies) not on crates.io at the required version:"
    printf '  %s\n' "${missing[@]}"
    echo "Publish them first (an earlier tier in .github/workflows/release-crates.yml), or bump them so the pin matches a published version."
    exit 1
fi
echo "${crate}: every workspace dependency is on crates.io"
