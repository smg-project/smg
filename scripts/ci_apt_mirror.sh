#!/bin/bash
# Point apt at the first reachable Ubuntu mirror before any apt-get runs.
#
# When Canonical's main archive front-ends are unreachable, every CI lane that
# installs a package fails or crawls: apt waits out its timeout on each
# address, setup steps blow through the job budget, and a retry helper cannot
# help because the host itself is down. Canonical also serves the same signed
# archive from mirrors inside each cloud, on separate infrastructure. apt
# verifies the signed release files and package hashes whatever host serves
# them, so switching hosts changes nothing about what gets installed.
#
# Usage: bash ci_apt_mirror.sh
#
# Idempotent: call it right before each `apt-get update`. It probes the
# mirrors in order, rewrites the Ubuntu sources to the first one that answers,
# and writes an apt.conf snippet that skips IPv6 (the runner pods have no IPv6
# route) and fails a dead host in seconds instead of minutes.
#
# Environment:
#   CI_APT_MIRRORS  space-separated base URLs tried in order. Default: the
#                   Canonical-run mirrors for OCI Frankfurt, Azure and AWS
#                   Frankfurt, then the main archive.
#   CI_APT_ROOT     apt configuration root (default /etc/apt).
#   CI_OS_RELEASE   os-release file (default /etc/os-release).

set -euo pipefail

APT_ROOT="${CI_APT_ROOT:-/etc/apt}"
OS_RELEASE="${CI_OS_RELEASE:-/etc/os-release}"
CONF="${APT_ROOT}/apt.conf.d/99-ci-mirror"
MAIN_ARCHIVE="http://archive.ubuntu.com/ubuntu"
DEFAULT_MIRRORS="http://eu-frankfurt-1.clouds.archive.ubuntu.com/ubuntu http://azure.archive.ubuntu.com/ubuntu http://eu-central-1.ec2.archive.ubuntu.com/ubuntu ${MAIN_ARCHIVE}"
read -ra MIRRORS <<<"${CI_APT_MIRRORS:-${DEFAULT_MIRRORS}}"

log() { echo "ci_apt_mirror: $*"; }

if [ ! -d "${APT_ROOT}/apt.conf.d" ]; then
    log "no ${APT_ROOT}/apt.conf.d; nothing to do"
    exit 0
fi
os_id=""
codename=""
if [ -r "${OS_RELEASE}" ]; then
    os_id="$(. "${OS_RELEASE}" && echo "${ID:-}")"
    codename="$(. "${OS_RELEASE}" && echo "${VERSION_CODENAME:-}")"
fi
if [ "${os_id}" != "ubuntu" ] || [ -z "${codename}" ]; then
    log "not an Ubuntu host (ID='${os_id}'); leaving apt sources alone"
    exit 0
fi

# An earlier call in this job already settled it, unless no mirror answered
# then: re-probe in that case, since the outage may have cleared.
if [ -f "${CONF}" ] && ! grep -q "mirror=none" "${CONF}"; then
    log "already configured ($(grep -o 'mirror=[^ )]*' "${CONF}" | head -n 1))"
    exit 0
fi

SUDO=""
if [ ! -w "${APT_ROOT}/apt.conf.d" ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        log "cannot write ${APT_ROOT} and no sudo; leaving apt sources alone"
        exit 0
    fi
fi

reachable() {
    local url="$1" host
    if command -v curl >/dev/null 2>&1; then
        curl -4 -fsSL --max-time 10 -o /dev/null "${url}/dists/${codename}/InRelease"
        return
    fi
    # A bare base image (docker build bootstrap) has no curl yet; settle for
    # a TCP connect, which is exactly what fails when a front-end is down.
    host="${url#*://}"
    host="${host%%/*}"
    timeout 10 bash -c "exec 3<>/dev/tcp/${host}/80" 2>/dev/null
}

chosen=""
for mirror in "${MIRRORS[@]}"; do
    if reachable "${mirror}"; then
        chosen="${mirror}"
        break
    fi
    log "${mirror} did not answer"
done

# Written even when nothing answered: the timeouts still stop a dead host from
# eating the job budget.
${SUDO} tee "${CONF}" >/dev/null <<CONF_EOF
// Written by scripts/ci_apt_mirror.sh (mirror=${chosen:-none}).
// The runner pods have no IPv6 route, so v6 addresses only add a failed
// connect per fetch. Short timeouts fail a dead host in seconds.
Acquire::ForceIPv4 "true";
Acquire::http::Timeout "20";
Acquire::Retries "2";
CONF_EOF

if [ -z "${chosen}" ]; then
    log "no mirror answered; leaving apt sources alone"
    exit 0
fi
if [ "${chosen}" = "${MAIN_ARCHIVE}" ]; then
    log "main archive answers; leaving apt sources alone"
    exit 0
fi

# Both the deb822 (*.sources) and the legacy (sources.list, *.list) forms.
# security.ubuntu.com is rewritten too: the cloud mirrors carry the security
# pocket, and it fails together with the main archive.
pattern='https?://([a-z]{2}\.)?(archive|security)\.ubuntu\.com/ubuntu/?'
rewritten=0
for file in "${APT_ROOT}/sources.list" "${APT_ROOT}"/sources.list.d/*.list "${APT_ROOT}"/sources.list.d/*.sources; do
    [ -f "${file}" ] || continue
    grep -Eq "${pattern}" "${file}" || continue
    tmp="$(mktemp)"
    sed -E "s#${pattern}#${chosen}/#g" "${file}" >"${tmp}"
    ${SUDO} cp "${tmp}" "${file}"
    rm -f "${tmp}"
    rewritten=$((rewritten + 1))
    log "rewrote ${file}"
done
log "using ${chosen} (${rewritten} source file(s) rewritten)"
