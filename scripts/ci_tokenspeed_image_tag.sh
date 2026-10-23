#!/bin/bash
# Print the carrier tag of the pinned TokenSpeed nightly, or the version itself
# with --version. The pin is .github/versions/tokenspeed-nightly.txt: the
# carrier is built from it, the e2e lanes resolve its tag and install it, and a
# bump is a pull request gated by its own TokenSpeed lanes, so a nightly that
# does not import never reaches the lanes of every other pull request.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="${TOKENSPEED_NIGHTLY_PIN:-${SCRIPT_DIR}/../.github/versions/tokenspeed-nightly.txt}"

version="$(tr -d '[:space:]' < "$PIN")"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.post[0-9]{8}$ ]]; then
    echo "ERROR: ${PIN} must hold one dated TokenSpeed nightly version (x.y.z.postYYYYMMDD), got '${version}'" >&2
    exit 1
fi
if [ "${1:-}" = "--version" ]; then
    printf '%s\n' "$version"
else
    printf 'ci-tokenspeed-%s\n' "${version##*post}"
fi
