#!/bin/bash
# Fetch the prebuilt TokenSpeed payload onto the bare runner, if available.
#
# The GPU e2e jobs run directly on the runner pods (no job container), so the
# prebuilt CI image (docker/ci-tokenspeed.Dockerfile) is consumed as a
# CARRIER: pull it, create a stopped container, and docker-cp the baked
# payload out:
#   /opt/smg-ci          venv + prebuilt stamp
#   /opt/tokenspeed-src  checkout the venv's editable installs point at
# then advertise the venv via SMG_BAKED_VENV (GITHUB_ENV, i.e. subsequent
# steps) so ci_setup_python_venv.sh adopts it and
# ci_install_tokenspeed.sh's stamp check skips the source build.
#
# PER-NODE CACHE. The runner pods are ephemeral and their dind storage is a
# per-pod emptyDir, so without a cache every lane re-pulls the ~8 GB carrier
# over a single ~25 MB/s registry connection (5+ min) and docker-cp's the
# payload out again (2+ min). The extracted payload is therefore kept on the
# node's NVMe hostPath mount (/models, shared by every runner pod on the
# node), keyed by the content-addressed image tag:
#   $CACHE_ROOT/<tag>/{smg-ci,tokenspeed-src}     populated once per node per tag
#   $CACHE_ROOT/<tag>/.complete                   written last; marks it usable
#   $CACHE_ROOT/jobs/<run>-<job>-<attempt>.XXXXXX/  this job's private copy
# Every job gets its own copy because the lane writes into the venv (the PR's
# smg wheel, the grpc servicer glue) and Python drops bytecode into the source
# tree. The copy is reflink-when-possible on the same filesystem, so it is
# cheap; the fixed /opt paths become symlinks into it. e2e-gpu-job.yml removes
# the job copy when the job ends (TOKENSPEED_PREBUILT_JOB_DIR); copies left by
# hard-killed jobs are swept here after 24 h, superseded tags after 7 days.
# An entry is trusted only when its marker AND the files the payload needs
# are present (a killed populate or invalidation can leave a marker over a
# hollow tree), and a populate retires whatever is at the entry path by
# rename-then-delete before moving its own tree in, so a damaged entry can
# never poison later lanes. Without a writable shared cache root the same
# flow runs against a job-local root under RUNNER_TEMP: no reuse, but still
# the ~8 min carrier path rather than the ~20 min source build; the payload
# is then moved into the job dir rather than copied, since it has one reader.
#
# TOLERANT BY DESIGN: this script must never fail the job. No image resolved,
# no docker on the runner, no writable cache, a failed pull or extraction —
# each just means the lane falls back to the source build, with a log line
# saying why.

set -uo pipefail # deliberately NOT -e: every failure is a soft fallback

IMAGE="${TOKENSPEED_PREBUILT_IMAGE:-}"
CACHE_ROOT="${TOKENSPEED_PREBUILT_CACHE_ROOT:-/models/.ci-cache/tokenspeed}"
INSTALL_ROOT="${TOKENSPEED_PREBUILT_INSTALL_ROOT:-/opt}"

log() { echo "[fetch-tokenspeed-prebuilt] $*"; }
fallback() {
    log "$*; lane will build from source"
    exit 0
}

[ -n "$IMAGE" ] || fallback "TOKENSPEED_PREBUILT_IMAGE unset"
command -v docker &> /dev/null || fallback "docker not available on this runner"
if command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi

# The tag is content-addressed (scripts/ci_tokenspeed_image_tag.sh), so it is
# the cache key; reduce it to a plain path segment.
tag="$(printf '%s' "${IMAGE##*:}" | tr -c 'A-Za-z0-9._-' '_')"
[ -n "$tag" ] || fallback "cannot derive a tag from ${IMAGE}"
init_cache_root() { mkdir -p "$1/.locks" "$1/jobs" 2> /dev/null; }
job_local=0
if ! init_cache_root "$CACHE_ROOT"; then
    log "cache root ${CACHE_ROOT} not writable; using a job-local cache instead (no reuse across lanes)"
    CACHE_ROOT="${RUNNER_TEMP:-/tmp}/tokenspeed-prebuilt-cache"
    init_cache_root "$CACHE_ROOT" || fallback "no writable cache root"
    job_local=1
fi
entry="${CACHE_ROOT}/${tag}"
# The marker alone is not proof, and neither are the dirs (an interrupted
# rm -rf leaves them hollow): require the files the payload actually needs.
entry_usable() {
    [ -f "$entry/.complete" ] && [ -f "$entry/smg-ci/tokenspeed.ref" ] \
        && [ -x "$entry/smg-ci/.venv/bin/python" ] && [ -d "$entry/tokenspeed-src" ]
}
# Invalidate atomically: rename first (instant), then delete under a name
# nothing consults, so a kill mid-delete cannot leave a half-emptied entry.
retire_entry() {
    [ -e "$entry" ] || return 0
    local old
    old="$(mktemp -d "${CACHE_ROOT}/.${tag}.old.XXXXXX")" && mv -T "$entry" "$old" && rm -rf "$old"
}

# Housekeeping: job copies left behind by cancelled jobs, and tags nothing
# pulls anymore (the tag changes on every tokenspeed.ref bump).
find "${CACHE_ROOT}/jobs" -mindepth 1 -maxdepth 1 -type d -mmin +1440 -exec rm -rf {} + 2> /dev/null
find "${CACHE_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name "${tag}" ! -name jobs ! -name .locks \
    -mtime +7 -exec rm -rf {} + 2> /dev/null

pull_image() {
    docker pull "$IMAGE" && return 0
    # A private GHCR package rejects anonymous pulls; retry authenticated
    # when the workflow token is available.
    if [ -n "${GITHUB_TOKEN:-}" ] \
        && docker login ghcr.io -u "${GITHUB_ACTOR:-github-actions}" --password-stdin <<< "$GITHUB_TOKEN" > /dev/null 2>&1 \
        && docker pull "$IMAGE"; then
        log "pull succeeded after ghcr login"
        return 0
    fi
    return 1
}

# Runs under the per-tag lock. Extracts into a sibling temp dir and renames it
# into place, so a reader never sees a half-written entry and a failure leaves
# nothing behind. Any leftover at the entry path is retired first; `mv -T`
# then cannot nest the temp dir inside a surviving directory.
populate() {
    pull_image || return 1
    local cid tmp
    cid="$(docker create "$IMAGE")" || return 1
    [ -n "$cid" ] || return 1
    tmp="$(mktemp -d "${CACHE_ROOT}/.${tag}.tmp.XXXXXX")" || {
        docker rm -f "$cid" > /dev/null 2>&1
        return 1
    }
    # The payload was baked as root; the job user needs write access for the
    # per-PR glue installs, so repair ownership before the entry goes live.
    if docker cp "$cid:/opt/smg-ci" "$tmp/smg-ci" \
        && docker cp "$cid:/opt/tokenspeed-src" "$tmp/tokenspeed-src" \
        && $SUDO chown -R "$(id -u):$(id -g)" "$tmp" \
        && retire_entry \
        && mv -T "$tmp" "$entry" \
        && touch "$entry/.complete"; then
        docker rm -f "$cid" > /dev/null 2>&1 || true
        return 0
    fi
    docker rm -f "$cid" > /dev/null 2>&1 || true
    rm -rf "$tmp"
    return 1
}

if entry_usable; then
    log "cache hit: ${entry}"
else
    (
        flock -w 1800 200 || exit 1
        if entry_usable; then
            log "cache hit (populated by another job while waiting): ${entry}"
            exit 0
        fi
        log "cache miss; pulling ${IMAGE}..."
        populate
    ) 200> "${CACHE_ROOT}/.locks/${tag}.lock" || fallback "could not populate ${entry}"
fi
# Keep an in-use tag out of the 7-day sweep.
touch "$entry"

# Unique by construction: runner pods have their own PID namespaces, so two
# lanes of one run on the same node can share run id, job, attempt and pid.
job="$(mktemp -d "${CACHE_ROOT}/jobs/${GITHUB_RUN_ID:-local}-${GITHUB_JOB:-job}-${GITHUB_RUN_ATTEMPT:-1}.XXXXXX")" \
    || fallback "could not create a job dir under ${CACHE_ROOT}/jobs"
if [ "$job_local" = 1 ]; then
    # Pod-local storage has no reflink and the entry has exactly one reader:
    # move it (a rename) rather than hold two ~15 GB trees on the pod.
    transfer=(mv -T)
else
    transfer=(cp -a --reflink=auto)
fi
if ! { "${transfer[@]}" "$entry/smg-ci" "$job/smg-ci" \
    && "${transfer[@]}" "$entry/tokenspeed-src" "$job/tokenspeed-src"; }; then
    rm -rf "$job"
    fallback "could not copy the payload for this job"
fi

# Undo a half-done install (a dangling symlink would otherwise greet the
# source build) and the job copy, then fall back.
abort_install() {
    $SUDO rm -rf "${INSTALL_ROOT}/smg-ci" "${INSTALL_ROOT}/tokenspeed-src" > /dev/null 2>&1 || true
    rm -rf "$job"
    fallback "$@"
}

# Fixed destination paths (the venv's shebangs and editable installs are
# absolute) now point into the job's copy.
if ! { $SUDO rm -rf "${INSTALL_ROOT}/smg-ci" "${INSTALL_ROOT}/tokenspeed-src" \
    && $SUDO ln -s "$job/smg-ci" "${INSTALL_ROOT}/smg-ci" \
    && $SUDO ln -s "$job/tokenspeed-src" "${INSTALL_ROOT}/tokenspeed-src"; }; then
    abort_install "install to ${INSTALL_ROOT} failed"
fi

# Later steps only learn about the payload through GITHUB_ENV; if that write
# fails the install is useless and its copy would linger, so roll it back.
if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "SMG_BAKED_VENV=${INSTALL_ROOT}/smg-ci/.venv"
        echo "TOKENSPEED_PREBUILT_JOB_DIR=${job}"
    } >> "$GITHUB_ENV" || abort_install "could not write ${GITHUB_ENV}"
fi
log "Prebuilt payload installed (stamp: $(cat "${INSTALL_ROOT}/smg-ci/tokenspeed.ref" 2> /dev/null || echo missing))"
