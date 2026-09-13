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
# and writes an apt.conf snippet that fails a dead host in seconds instead of
# minutes.
#
# Environment:
#   CI_APT_MIRRORS  space-separated base URLs tried in order. Default: the
#                   Canonical-run mirrors for OCI Frankfurt, Azure and AWS
#                   Frankfurt, then the main archive. A candidate must serve
#                   both <codename> and <codename>-security; the security
#                   lines are routed through it too.
#   CI_APT_ROOT     apt configuration root (default /etc/apt).
#   CI_OS_RELEASE   os-release file (default /etc/os-release).

set -euo pipefail

APT_ROOT="${CI_APT_ROOT:-/etc/apt}"
OS_RELEASE="${CI_OS_RELEASE:-/etc/os-release}"
CONF="${APT_ROOT}/apt.conf.d/99-ci-mirror"
MAIN_ARCHIVE="http://archive.ubuntu.com/ubuntu"
SECURITY_ARCHIVE="http://security.ubuntu.com/ubuntu"
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
    local url="$1" suite="${2:-${codename}}" rest host base status
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --max-time 10 -o /dev/null "${url}/dists/${suite}/InRelease"
        return
    fi
    # A bare base image (docker build bootstrap) has no curl yet: speak just
    # enough HTTP over bash's /dev/tcp to check the suite is really served.
    case "${url}" in
        http://*) ;;
        *) log "cannot probe ${url} without curl"; return 1 ;;
    esac
    rest="${url#http://}"
    host="${rest%%/*}"
    base="${rest#"${host}"}"
    status="$(timeout 10 bash -c '
        exec 3<>"/dev/tcp/$1/80" || exit 1
        printf "HEAD %s HTTP/1.0\r\nHost: %s\r\n\r\n" "$2" "$1" >&3
        IFS= read -r line <&3 && printf "%s" "${line}"
    ' _ "${host}" "${base}/dists/${suite}/InRelease" 2>/dev/null)" || return 1
    case "${status}" in
        *" 200 "*) return 0 ;;
        *) return 1 ;;
    esac
}

# Matches the main archive, its country aliases (us.archive...) and the cloud
# mirrors (azure.archive..., *.clouds.archive..., *.ec2.archive...), so a host
# that already carries a mirror, including one this script chose earlier, is
# moved again if that mirror stops answering. ports.ubuntu.com (arm64) is
# left alone: the x86 mirrors do not carry ubuntu-ports.
pattern='https?://([a-z0-9-]+\.)*(archive|security)\.ubuntu\.com/ubuntu/?'
source_files=("${APT_ROOT}/sources.list" "${APT_ROOT}"/sources.list.d/*.list "${APT_ROOT}"/sources.list.d/*.sources)

# A host already on a Canonical mirror (GitHub-hosted runners use the Azure
# one) keeps it while it answers; only a dead mirror is replaced. The rewrite
# below still runs for it, so any line left on another host joins it.
current="$(cat "${source_files[@]}" 2>/dev/null | grep -Eo "${pattern}" | grep -vE '://(archive|security)\.ubuntu\.com/' | head -n 1 || true)"
current="${current%/}"

# The rewrite folds the security lines into the chosen host, so it has to
# serve the security pocket as well as the release.
serves_both() {
    reachable "$1" || { log "$1 did not answer for ${codename}"; return 1; }
    reachable "$1" "${codename}-security" || { log "$1 did not answer for ${codename}-security"; return 1; }
}

chosen=""
if [ -n "${current}" ] && serves_both "${current}"; then
    chosen="${current}"
fi
for mirror in "${MIRRORS[@]}"; do
    [ -z "${chosen}" ] || break
    [ "${mirror%/}" != "${current}" ] || continue
    if serves_both "${mirror}"; then
        chosen="${mirror}"
    fi
done

# The marker records the outcome; "none" makes the next call probe again.
write_conf() {
    if ! ${SUDO} tee "${CONF}" >/dev/null <<CONF_EOF
// Written by scripts/ci_apt_mirror.sh (mirror=$1).
// Short timeouts fail a dead host in seconds instead of minutes.
Acquire::http::Timeout "20";
Acquire::https::Timeout "20";
Acquire::Retries "2";
CONF_EOF
    then
        log "could not write ${CONF}"
    fi
}

if [ -z "${chosen}" ]; then
    write_conf none
    log "no mirror answered; leaving apt sources alone"
    exit 0
fi
# Sources already on the main archive stay as they are, provided the
# separate security origin answers too; if it does not, the rewrite below
# moves the security lines onto the main archive, which serves that pocket.
if [ "${chosen}" = "${MAIN_ARCHIVE}" ] && [ -z "${current}" ] \
    && reachable "${SECURITY_ARCHIVE}" "${codename}-security"; then
    write_conf "${chosen}"
    log "main archive answers; leaving apt sources alone"
    exit 0
fi
# Both the deb822 (*.sources) and the legacy (sources.list, *.list) forms.
# security.ubuntu.com is rewritten too: the cloud mirrors carry the security
# pocket, and it fails together with the main archive.
matched=0
rewritten=0
failed=0
for file in "${source_files[@]}"; do
    [ -f "${file}" ] || continue
    grep -Eq "${pattern}" "${file}" || continue
    matched=$((matched + 1))
    # Stage the copy next to the file so the final rename is atomic: apt
    # never sees a partial file, whatever fails along the way.
    if ! staged="$(${SUDO} mktemp "$(dirname "${file}")/.ci-apt-mirror.XXXXXX")"; then
        log "could not stage a copy of ${file}; leaving it alone"
        failed=$((failed + 1))
        continue
    fi
    if ! sed -E "s#${pattern}#${chosen}/#g" "${file}" | ${SUDO} tee "${staged}" >/dev/null; then
        log "could not rewrite ${file}; leaving it alone"
        ${SUDO} rm -f "${staged}"
        failed=$((failed + 1))
        continue
    fi
    if ${SUDO} cmp -s "${staged}" "${file}"; then
        ${SUDO} rm -f "${staged}"
        continue
    fi
    if ! ${SUDO} chmod 644 "${staged}" || ! ${SUDO} mv -f "${staged}" "${file}"; then
        log "could not write ${file}; leaving it alone"
        ${SUDO} rm -f "${staged}"
        failed=$((failed + 1))
        continue
    fi
    rewritten=$((rewritten + 1))
    log "rewrote ${file}"
done
if [ "${failed}" -gt 0 ]; then
    write_conf none
    log "${failed} source file(s) could not be rewritten; the next call will try again"
elif [ "${matched}" -eq 0 ]; then
    write_conf unmatched
    log "chose ${chosen} but no Ubuntu source lines matched; apt sources left as they were"
else
    write_conf "${chosen}"
    log "using ${chosen} (${rewritten} source file(s) rewritten)"
fi
