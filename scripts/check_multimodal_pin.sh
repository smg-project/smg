#!/usr/bin/env bash
# The workspace builds llm-multimodal from a git rev pin in the root
# Cargo.toml, while crates/multimodal is a source-only submodule mirror for
# browsing. Fail if the two point at different commits: drifting mirrors
# show developers code that is not what the build compiles.
set -euo pipefail
cd "$(dirname "$0")/.."

rev=$(grep -oP '^llm-multimodal = \{[^}]*rev = "\K[0-9a-f]{40}' Cargo.toml) || {
    echo "::error::no llm-multimodal git rev pin found in the root Cargo.toml"
    exit 1
}
gitlink=$(git ls-files -s crates/multimodal | awk '$1 == "160000" { print $2 }') || true
if [ -z "$gitlink" ]; then
    echo "::error::crates/multimodal is not a git submodule (gitlink missing)"
    exit 1
fi
if [ "$rev" != "$gitlink" ]; then
    echo "::error::llm-multimodal pin drift: Cargo.toml rev $rev, submodule gitlink $gitlink"
    echo "Update both in the same commit:"
    echo "  git -C crates/multimodal fetch origin && git -C crates/multimodal checkout $rev"
    exit 1
fi
echo "llm-multimodal pin and submodule mirror agree at $rev"
