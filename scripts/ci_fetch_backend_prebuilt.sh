#!/bin/bash
# Fetch a prebuilt GPU backend carrier payload (vllm, sglang, or trtllm) onto
# a bare runner pod, if available.
#
# When the runner pod already carries a matching prepared environment under
# /opt/smg-ci/<backend>/.venv (e.g. built from docker/ci-gpu-runner.Dockerfile
# target `runner`), this script exits immediately with zero Docker work.
#
# On bare k8s runner pods, the per-backend carrier image published by
# ci-gpu-runner-image.yml uses an immutable content-addressed tag
# (`ci-gpu-backend-<backend>-<sha256>`). A warm node hits
# `/models/.ci-cache/<backend>/<tag>` immediately with zero registry or Docker
# calls. On a cache miss, the carrier is pulled once per node under a per-tag
# flock. Each job gets its own reflink copy of `/opt/smg-ci/<backend>`
# (symlinked into place so `ci_setup_python_venv.sh` adopts it and per-PR
# installs stay isolated). When the carrier also includes `/usr/local/cuda-13.0`
# (sglang, trtllm), that read-only toolkit is symlinked directly from the
# shared cache with zero per-job copy.
#
# Usage: ci_fetch_backend_prebuilt.sh <vllm|sglang|trtllm>
#
# TOLERANT BY DESIGN: never fails the job. Missing image, no docker, unwritable
# cache, or failed pull/extraction logs the reason and exits 0 so the backend
# installer falls back to a fresh environment.

set -uo pipefail # deliberately NOT -e: every failure is a soft fallback

BACKEND="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ci_prepared_backend_env.sh"

log() { echo "[fetch-${BACKEND:-backend}-prebuilt] $*"; }
fallback() {
    log "$*; lane will use fresh environment setup"
    exit 0
}

case "$BACKEND" in
    vllm|sglang|trtllm) ;;
    *)
        log "unsupported backend '${BACKEND}'"
        exit 0
        ;;
esac

PREINSTALLED_ROOT="${SMG_PREINSTALLED_ROOT:-/opt/smg-ci}"
CACHE_ROOT="${SMG_PREBUILT_CACHE_ROOT:-/models/.ci-cache/${BACKEND}}"
CUDA_INSTALL_ROOT="${SMG_PREBUILT_CUDA_ROOT:-/usr/local}"
IMAGE="${SMG_BACKEND_PREBUILT_IMAGE:-}"
DEFAULT_CARRIER_BASE="ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55"

# Fast path: runner image already bakes a matching prepared environment.
if smg_prepared_env_matches "$BACKEND" "${PREINSTALLED_ROOT}/${BACKEND}/.venv" 2> /dev/null; then
    log "Prepared ${BACKEND} environment already present at ${PREINSTALLED_ROOT}/${BACKEND}/.venv"
    exit 0
fi

if [ "$(id -u)" != 0 ] && command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi

# Resolve the immutable content-addressed carrier image tag when the caller did
# not override SMG_BACKEND_PREBUILT_IMAGE. Because the tag hashes every build
# input, warm nodes can hit the local NVMe cache without any registry call.
if [ -z "$IMAGE" ]; then
    carrier_base="${SMG_CI_GPU_CARRIER_BASE_IMAGE:-$DEFAULT_CARRIER_BASE}"
    driver_ver="${SMG_CI_GPU_RUNNER_CUDA_DRIVER_VERSION:-580.65.06}"
    repo_owner="${GITHUB_REPOSITORY_OWNER:-smg-project}"
    image_repo="${SMG_CI_GPU_RUNNER_IMAGE_REPOSITORY:-ghcr.io/${repo_owner}/smg}"
    resolved_tag="$(python3 "${SCRIPT_DIR}/ci_gpu_runner_image.py" tag \
        --backend "$BACKEND" \
        --base-image "$carrier_base" \
        --cuda-driver-version "$driver_ver" 2> /dev/null || true)"
    [ -n "$resolved_tag" ] || fallback "could not compute carrier tag for ${BACKEND}"
    IMAGE="${image_repo}:${resolved_tag}"
fi

log "Prebuilt image: ${IMAGE}"

# Key the cache by the complete image reference so identical tags across
# different override repositories never collide, while keeping a readable tag
# prefix for diagnostics.
tag_part="$(printf '%s' "${IMAGE##*:}" | tr -c 'A-Za-z0-9._-' '_')"
image_hash="$(printf '%s' "$IMAGE" | sha256sum | cut -c 1-12)"
tag="${tag_part}-${image_hash}"
[ -n "$tag_part" ] && [ -n "$image_hash" ] || fallback "cannot derive a cache key from ${IMAGE}"

init_cache_root() { mkdir -p "$1/.locks" "$1/jobs" 2> /dev/null; }
job_local=0
if ! init_cache_root "$CACHE_ROOT"; then
    log "cache root ${CACHE_ROOT} not writable; using a job-local cache instead (no reuse across lanes)"
    CACHE_ROOT="${RUNNER_TEMP:-/tmp}/${BACKEND}-prebuilt-cache"
    init_cache_root "$CACHE_ROOT" || fallback "no writable cache root"
    job_local=1
fi

entry="${CACHE_ROOT}/${tag}"
entry_usable() {
    [ -f "$entry/.complete" ] \
        && [ -f "$entry/smg-ci/${BACKEND}/.venv/.smg-ci-recipe" ] \
        && [ -x "$entry/smg-ci/${BACKEND}/.venv/bin/python" ]
}
retire_entry() {
    [ -e "$entry" ] || return 0
    local old
    old="$(mktemp -d "${CACHE_ROOT}/.${tag}.old.XXXXXX")" && mv -T "$entry" "$old" && rm -rf "$old"
}

# Housekeeping: job copies left by cancelled jobs (>24 h) and superseded tags (>7 d).
find "${CACHE_ROOT}/jobs" -mindepth 1 -maxdepth 1 -type d -mmin +1440 -exec rm -rf {} + 2> /dev/null
find "${CACHE_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name "${tag}" ! -name jobs ! -name .locks \
    -mtime +7 -exec rm -rf {} + 2> /dev/null

pull_image() {
    command -v docker &> /dev/null || return 1
    docker pull "$IMAGE" && return 0
    if [ -n "${GITHUB_TOKEN:-}" ] \
        && docker login ghcr.io -u "${GITHUB_ACTOR:-github-actions}" --password-stdin <<< "$GITHUB_TOKEN" > /dev/null 2>&1 \
        && docker pull "$IMAGE"; then
        log "pull succeeded after ghcr login"
        return 0
    fi
    return 1
}

populate() {
    pull_image || return 1
    local cid tmp
    cid="$(docker create "$IMAGE")" || return 1
    [ -n "$cid" ] || return 1
    tmp="$(mktemp -d "${CACHE_ROOT}/.${tag}.tmp.XXXXXX")" || {
        docker rm -f "$cid" > /dev/null 2>&1
        return 1
    }
    mkdir -p "$tmp/smg-ci"
    if docker cp "$cid:/opt/smg-ci/${BACKEND}" "$tmp/smg-ci/${BACKEND}"; then
        # Extract CUDA 13.0 toolkit when baked into the carrier (sglang, trtllm).
        docker cp "$cid:/usr/local/cuda-13.0" "$tmp/cuda-13.0" > /dev/null 2>&1 || rm -rf "$tmp/cuda-13.0"
        if $SUDO chown -R "$(id -u):$(id -g)" "$tmp" \
            && retire_entry \
            && mv -T "$tmp" "$entry" \
            && touch "$entry/.complete"; then
            docker rm -f "$cid" > /dev/null 2>&1 || true
            return 0
        fi
    fi
    docker rm -f "$cid" > /dev/null 2>&1 || true
    rm -rf "$tmp"
    return 1
}

if entry_usable; then
    log "cache hit: ${entry}"
else
    command -v docker &> /dev/null || fallback "docker not available on this runner"
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
touch "$entry"

job="$(mktemp -d "${CACHE_ROOT}/jobs/${GITHUB_RUN_ID:-local}-${GITHUB_JOB:-job}-${GITHUB_RUN_ATTEMPT:-1}.XXXXXX")" \
    || fallback "could not create a job dir under ${CACHE_ROOT}/jobs"
if [ "$job_local" = 1 ]; then
    transfer=(mv -T)
else
    transfer=(cp -a --reflink=auto)
fi
if ! "${transfer[@]}" "$entry/smg-ci/${BACKEND}" "$job/${BACKEND}"; then
    rm -rf "$job"
    fallback "could not copy the payload for this job"
fi

cuda_src=""
if [ -x "$entry/cuda-13.0/bin/nvcc" ]; then
    if [ "$job_local" = 1 ]; then
        if mv -T "$entry/cuda-13.0" "$job/cuda-13.0"; then
            cuda_src="$job/cuda-13.0"
        fi
    else
        cuda_src="$entry/cuda-13.0"
    fi
fi

abort_install() {
    $SUDO rm -rf "${PREINSTALLED_ROOT}/${BACKEND}" > /dev/null 2>&1 || true
    rm -rf "$job"
    fallback "$@"
}

if ! { $SUDO mkdir -p "${PREINSTALLED_ROOT}" \
    && $SUDO rm -rf "${PREINSTALLED_ROOT}/${BACKEND}" \
    && $SUDO ln -s "$job/${BACKEND}" "${PREINSTALLED_ROOT}/${BACKEND}"; }; then
    abort_install "install to ${PREINSTALLED_ROOT}/${BACKEND} failed"
fi

# Link read-only CUDA 13.0 toolkit directly from the shared NVMe cache when the
# runner lacks nvcc 13. Zero per-job copy.
if [ -n "$cuda_src" ] && [ ! -x "${CUDA_INSTALL_ROOT}/cuda-13.0/bin/nvcc" ]; then
    if $SUDO mkdir -p "${CUDA_INSTALL_ROOT}" \
        && $SUDO rm -rf "${CUDA_INSTALL_ROOT}/cuda-13.0" \
        && $SUDO ln -s "$cuda_src" "${CUDA_INSTALL_ROOT}/cuda-13.0"; then
        if [ ! -x "${CUDA_INSTALL_ROOT}/cuda/bin/nvcc" ]; then
            $SUDO ln -sfn "${CUDA_INSTALL_ROOT}/cuda-13.0" "${CUDA_INSTALL_ROOT}/cuda" || true
        fi
        log "Linked cached CUDA 13.0 toolkit from ${cuda_src}"
    else
        log "WARNING: could not link cached CUDA 13.0 toolkit; installer may apt-install it"
    fi
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SMG_PREBUILT_JOB_DIR=${job}" >> "$GITHUB_ENV" || abort_install "could not write ${GITHUB_ENV}"
fi
log "Prebuilt ${BACKEND} payload installed at ${PREINSTALLED_ROOT}/${BACKEND}"
