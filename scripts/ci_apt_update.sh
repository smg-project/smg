#!/bin/bash
# Refresh apt's package indexes for a CI install step, and survive a
# third-party source that does not answer.
#
# `apt-get update` exits 100 when any configured source fails, and the runner
# images carry sources the CI scripts never install from (a PPA for a newer
# git, the GPU and container vendors' repositories). When one of those is
# down, every job whose setup installs a package from the Ubuntu archive fails
# before it builds anything: a PPA mirror that answered 503 for an hour took
# the Rust lint and test jobs of a whole merge wave with it.
#
# Usage: ci_apt_update.sh [sudo]
#
# The optional word prefixes the apt-get commands. The first attempt is the
# update as configured; if it fails, the Ubuntu archive's own sources are
# refreshed alone (the lines and deb822 stanzas whose hosts are under
# ubuntu.com, which is where the install that follows takes its packages
# from), through the retry helper the other install scripts use, and a
# failure of that is the step's. The retry sits on the second stage on
# purpose: an outage of a source the step never installs from goes to the
# fallback at once, and a blip of the archive itself is retried where it
# matters. apt's own fetch retries (scripts/ci_apt_mirror.sh writes
# Acquire::Retries) still cover the briefest ones.
#
# A `mirror+file:<path>` URI (the form the GitHub-hosted images give the
# archive, listing its hosts in <path>) counts as the archive's when every
# host in the file is.
#
# Environment:
#   CI_APT_ROOT                 apt configuration root (default /etc/apt).
#   CI_APT_UPDATE_RETRY_DELAY   first delay of the retry, seconds (default 10).

set -euo pipefail

SUDO="${1:-}"
APT_ROOT="${CI_APT_ROOT:-/etc/apt}"
RETRY_DELAY="${CI_APT_UPDATE_RETRY_DELAY:-10}"
RETRY="$(dirname "${BASH_SOURCE[0]}")/ci_retry.sh"

log() { echo "ci_apt_update: $*" >&2; }

# shellcheck disable=SC2086 # an empty SUDO must vanish, not become an argument
if ${SUDO} apt-get update; then
    exit 0
fi
log "apt-get update failed; refreshing the Ubuntu archive's indexes alone"

ubuntu_host='https?://([a-z0-9-]+\.)*ubuntu\.com(/|[[:space:]]|$)'
staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT
parts="${staging}/sources.list.d"
mkdir "${parts}"
kept=0

# Whether one source URI is the archive's: a host under ubuntu.com, or a
# `mirror+file:<path>` whose every listed host (one per line, an optional
# priority word after it, `#` comments) is.
archive_uri() {
    local uri="$1" path hosts
    case "${uri}" in
        mirror+file:*)
            path="${uri#mirror+file:}"
            [ -r "${path}" ] || return 1
            hosts="$(sed -E 's/#.*//; s/^[[:space:]]+//; /^[[:space:]]*$/d' "${path}" | awk '{print $1}')"
            [ -n "${hosts}" ] || return 1
            ! printf '%s\n' "${hosts}" | grep -qvE "^${ubuntu_host}" ;;
        *)
            printf '%s\n' "${uri}" | grep -qE "^${ubuntu_host}" ;;
    esac
}

# The one-line form: `deb [options] URL suite components`, one source per
# line, in sources.list and in sources.list.d/*.list. Lines on other hosts are
# dropped; a file left without a source is not written.
keep_lines() {
    local source="$1" target="$2" line uri lines=""
    while IFS= read -r line; do
        uri="$(printf '%s\n' "${line}" | sed -E 's/^[[:space:]]*deb(-src)?[[:space:]]+(\[[^]]*\][[:space:]]+)?//' | awk '{print $1}')"
        archive_uri "${uri}" || continue
        lines+="${line}"$'\n'
    done < <(grep -E '^[[:space:]]*deb(-src)?[[:space:]]' "${source}" || true)
    [ -n "${lines}" ] || return 0
    printf '%s' "${lines}" >"${target}"
    kept=$((kept + 1))
}

legacy="${staging}/sources.list"
: >"${legacy}"
if [ -f "${APT_ROOT}/sources.list" ]; then
    keep_lines "${APT_ROOT}/sources.list" "${legacy}"
fi
for file in "${APT_ROOT}"/sources.list.d/*.list; do
    [ -f "${file}" ] || continue
    keep_lines "${file}" "${parts}/$(basename "${file}")"
done

# The deb822 form (sources.list.d/*.sources): a file is stanzas separated by
# blank lines, each with its own URIs field. A stanza is kept when every URI in
# it is the archive's, and a file is written with its kept stanzas: Canonical's
# ubuntu.sources keeps every stanza, a vendor's file none, a file that mixes an
# archive stanza with a vendor's keeps the archive stanza alone.
keep_deb822_stanza() {
    local stanza="$1" target="$2" uris uri
    # The URIs field's values are separated by any whitespace, newlines
    # included: a field may go on over indented continuation lines.
    uris="$(printf '%s' "${stanza}" | awk '
        tolower($0) ~ /^uris:/ { sub(/^[^:]*:/, ""); print; in_uris = 1; next }
        in_uris && /^[[:space:]]/ { print; next }
        { in_uris = 0 }' | tr '[:space:]' '\n' | sed '/^$/d' || true)"
    [ -n "${uris}" ] || return 0
    while IFS= read -r uri; do
        archive_uri "${uri}" || return 0
    done <<<"${uris}"
    if [ -s "${target}" ]; then
        printf '\n' >>"${target}"
    fi
    printf '%s' "${stanza}" >>"${target}"
}

keep_deb822() {
    local source="$1" target="$2" stanza="" line
    while IFS= read -r line || [ -n "${line}" ]; do
        if [ -z "${line//[[:space:]]/}" ]; then
            [ -z "${stanza}" ] || keep_deb822_stanza "${stanza}" "${target}"
            stanza=""
        else
            stanza+="${line}"$'\n'
        fi
    done <"${source}"
    [ -z "${stanza}" ] || keep_deb822_stanza "${stanza}" "${target}"
    if [ -s "${target}" ]; then
        kept=$((kept + 1))
    else
        rm -f "${target}"
    fi
}

for file in "${APT_ROOT}"/sources.list.d/*.sources; do
    [ -f "${file}" ] || continue
    keep_deb822 "${file}" "${parts}/$(basename "${file}")"
done

if [ "${kept}" -eq 0 ]; then
    log "no source under ubuntu.com found in ${APT_ROOT}; nothing to fall back to"
    exit 100
fi
log "${kept} Ubuntu source file(s) kept"
# List-Cleanup off: apt would otherwise delete the index files of the sources
# left out here, and on a runner that keeps its state a later step installing
# from one of them would find no package until an update of its own succeeds.
# They stay as the failed update left them, next to fresh Ubuntu indexes.
# shellcheck disable=SC2086
bash "${RETRY}" 3 "${RETRY_DELAY}" ${SUDO} apt-get update -o APT::Get::List-Cleanup=false \
    -o Dir::Etc::sourcelist="${legacy}" -o Dir::Etc::sourceparts="${parts}"
