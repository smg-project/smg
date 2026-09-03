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
# TOLERANT BY DESIGN: this script must never fail the job. No image resolved,
# no docker on the runner, a failed pull or extraction — each just means the
# lane falls back to the source build, with a log line saying why.

set -uo pipefail # deliberately NOT -e: every failure is a soft fallback

IMAGE="${TOKENSPEED_PREBUILT_IMAGE:-}"

log() { echo "[fetch-tokenspeed-prebuilt] $*"; }

if [ -z "$IMAGE" ]; then
    log "TOKENSPEED_PREBUILT_IMAGE unset; lane will build from source"
    exit 0
fi
if ! command -v docker &> /dev/null; then
    log "docker not available on this runner; lane will build from source"
    exit 0
fi
if command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi

log "Pulling ${IMAGE}..."
if ! docker pull "$IMAGE"; then
    # A private GHCR package rejects anonymous pulls; retry authenticated
    # when the workflow token is available.
    if [ -n "${GITHUB_TOKEN:-}" ] \
        && docker login ghcr.io -u "${GITHUB_ACTOR:-github-actions}" --password-stdin <<< "$GITHUB_TOKEN" > /dev/null 2>&1 \
        && docker pull "$IMAGE"; then
        log "pull succeeded after ghcr login"
    else
        log "pull failed; lane will build from source"
        exit 0
    fi
fi

cid="$(docker create "$IMAGE")"
if [ -z "$cid" ]; then
    log "docker create failed; lane will build from source"
    exit 0
fi
trap 'docker rm -f "$cid" > /dev/null 2>&1 || true' EXIT

staging="$(mktemp -d /tmp/tokenspeed-prebuilt.XXXXXX)"
if [ -z "$staging" ]; then
    log "mktemp failed; lane will build from source"
    exit 0
fi

if ! docker cp "$cid:/opt/smg-ci" "$staging/smg-ci" \
    || ! docker cp "$cid:/opt/tokenspeed-src" "$staging/tokenspeed-src"; then
    log "extraction failed; lane will build from source"
    rm -rf "$staging"
    exit 0
fi

# Fixed destination paths (the venv's shebangs and editable installs are
# absolute). Same-filesystem move, effectively instant.
if ! { $SUDO rm -rf /opt/smg-ci /opt/tokenspeed-src \
    && $SUDO mv "$staging/smg-ci" /opt/smg-ci \
    && $SUDO mv "$staging/tokenspeed-src" /opt/tokenspeed-src; }; then
    log "install to /opt failed; lane will build from source"
    rm -rf "$staging"
    exit 0
fi
rm -rf "$staging"

# The payload was baked as root; the job user needs write access for the
# per-PR glue installs (uv pip uninstall/install -e into the venv).
$SUDO chown -R "$(id -u):$(id -g)" /opt/smg-ci /opt/tokenspeed-src || true

if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SMG_BAKED_VENV=/opt/smg-ci/.venv" >> "$GITHUB_ENV"
fi
log "Prebuilt payload installed (stamp: $(cat /opt/smg-ci/tokenspeed.ref 2> /dev/null || echo missing))"
