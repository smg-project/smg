#!/bin/bash
# Install packages from the Ubuntu archive in a CI step, and move to another
# mirror when the one in use fails.
#
# A mirror can answer the mirror script's probe and still fail an install.
# One did for hours: a front-end served 404 for the release files, and
# another for a package that the index it had just served named. The index
# refresh and the install each retried against that same mirror, so every
# attempt read the same 404 and the job ended before any test.
#
# Usage: ci_apt_install.sh [--strict-mirror] [sudo] <apt-get install arguments>
# --strict-mirror makes a failed mirror probe fail the attempt (PostgreSQL setup).
#
# The optional first word prefixes the apt-get commands, as in
# ci_apt_update.sh. One attempt is the whole unit: the mirror choice
# (ci_apt_mirror.sh), the index refresh (ci_apt_update.sh) and the install.
# The unit runs through the retry helper, and an attempt that follows a failed
# one asks the mirror script for another mirror first (--reprobe), so it reads
# fresh lists from a different host.
#
# Environment:
#   CI_APT_INSTALL_RETRY_DELAY  first delay of the retry, seconds (default 10).

set -euo pipefail

here="$(dirname "${BASH_SOURCE[0]}")"

log() { echo "ci_apt_install: $*" >&2; }

strict_mirror=""
if [ "${1:-}" = "--strict-mirror" ]; then
    strict_mirror="--strict-mirror"
    shift
fi

if [ "${1:-}" = "--attempt" ]; then
    # One attempt of the unit. The file named next holds a line once an
    # attempt of this call has started: a later attempt finds it and re-probes.
    earlier="$2"
    shift 2
    SUDO=""
    if [ "${1:-}" = "sudo" ]; then
        SUDO="sudo"
        shift
    fi
    reprobe=""
    if [ -s "${earlier}" ]; then
        reprobe="--reprobe"
        log "an earlier attempt failed; asking for another mirror"
    fi
    echo started >"${earlier}"
    mirror_status=0
    # shellcheck disable=SC2086 # an empty word must vanish, not become an argument
    bash "${here}/ci_apt_mirror.sh" ${reprobe} || mirror_status=$?
    if [ "${mirror_status}" -ne 0 ]; then
        if [ -n "${strict_mirror}" ]; then
            exit "${mirror_status}"
        fi
        log "the mirror probe failed; the sources stay as they are"
    fi
    # The refresh's own fallback runs once: this unit is the retry, and its next
    # attempt reads another mirror where the fallback's would read the same one.
    # shellcheck disable=SC2086
    CI_APT_UPDATE_ATTEMPTS=1 bash "${here}/ci_apt_update.sh" ${SUDO}
    # shellcheck disable=SC2086
    exec ${SUDO} apt-get install "$@"
fi

if [ "$#" -eq 0 ] || { [ "$1" = "sudo" ] && [ "$#" -eq 1 ]; }; then
    echo "usage: ci_apt_install.sh [--strict-mirror] [sudo] <apt-get install arguments>" >&2
    exit 2
fi
earlier="$(mktemp)"
trap 'rm -f "${earlier}"' EXIT
bash "${here}/ci_retry.sh" 3 "${CI_APT_INSTALL_RETRY_DELAY:-10}" \
    bash "${BASH_SOURCE[0]}" ${strict_mirror} --attempt "${earlier}" "$@"
