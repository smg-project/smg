#!/bin/bash
# Print the content-addressed tag of the prebuilt TokenSpeed CI image.
#
# Single source of truth for the tag scheme, shared by
# .github/workflows/ci-tokenspeed-image.yml (build+push) and the
# detect-changes resolution in pr-test-rust.yml (pull) so the two can never
# drift.
#
# The tag covers the pinned engine ref AND a hash of the image tooling
# (Dockerfile + install/venv scripts + this script): a fix to any of them
# produces a new tag, so the image rebuilds on merge automatically instead
# of serving stale tooling behind an unchanged ref.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

ref="$(tr -d '[:space:]' < .github/versions/tokenspeed.ref)"
if [ -z "$ref" ]; then
    echo "ERROR: .github/versions/tokenspeed.ref is empty" >&2
    exit 1
fi

# sha256sum on Linux CI, shasum on macOS dev machines.
if command -v sha256sum &> /dev/null; then
    SHA256=(sha256sum)
else
    SHA256=(shasum -a 256)
fi

tooling_hash="$(cat \
    docker/ci-tokenspeed.Dockerfile \
    scripts/ci_install_tokenspeed.sh \
    scripts/ci_setup_python_venv.sh \
    scripts/ci_tokenspeed_image_tag.sh \
    | "${SHA256[@]}" | cut -c1-12)"

echo "ci-tokenspeed-${ref:0:12}-${tooling_hash}"
