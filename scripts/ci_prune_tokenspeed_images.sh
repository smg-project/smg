#!/bin/bash
# Delete superseded prebuilt TokenSpeed CI images from GHCR.
#
# The tags scripts/ci_tokenspeed_image_tag.sh prints are content-addressed
# (engine ref + a hash of the image tooling), so every bump strands the
# previous ~8 GB image under a tag nothing will ever ask for again. Keep the
# newest few and delete the rest.
#
# Keeping more than one is purely about speed. A run that resolved the
# previous tag before this ran still pulls it instead of falling back to the
# ~25 minute source build — and even when it can't,
# ci_fetch_tokenspeed_prebuilt.sh treats a failed pull as a soft fallback, so
# a deleted tag can never fail a lane.
#
# Only versions whose tags ALL carry the ci-tokenspeed- prefix are eligible.
# The package also holds the nightly and release images, and they are not this
# script's business.
#
# TOLERANT BY DESIGN: cleanup must never fail the publish it follows.
#
# Env knobs:
#   TOKENSPEED_IMAGE_KEEP  how many recent images to keep (default 2)
#   DRY_RUN=1              list what would be deleted, delete nothing

set -uo pipefail # deliberately NOT -e: every failure is a soft skip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEEP="${TOKENSPEED_IMAGE_KEEP:-2}"
OWNER="${GITHUB_REPOSITORY_OWNER:-smg-project}"
PACKAGE="${TOKENSPEED_IMAGE_PACKAGE:-smg}"
PREFIX="ci-tokenspeed-"

log() { echo "[prune-tokenspeed-images] $*"; }

if ! [[ "$KEEP" =~ ^[0-9]+$ ]] || [ "$KEEP" -lt 1 ]; then
    log "TOKENSPEED_IMAGE_KEEP=${KEEP} is not a positive integer; skipping"
    exit 0
fi
if ! command -v gh &> /dev/null; then
    log "gh CLI not available; skipping"
    exit 0
fi

# Never delete the tag the lanes are resolving right now, whatever its age.
current="$(bash "${SCRIPT_DIR}/ci_tokenspeed_image_tag.sh" 2> /dev/null)"
if [ -z "$current" ]; then
    log "could not resolve the current tag; skipping rather than guessing"
    exit 0
fi
log "current tag: ${current} (keeping newest ${KEEP})"

# Versions tagged exclusively with the ci-tokenspeed- prefix, newest first.
versions="$(gh api --paginate \
    "/orgs/${OWNER}/packages/container/${PACKAGE}/versions?per_page=100" \
    -q ".[]
        | select(.metadata.container.tags | length > 0)
        | select([.metadata.container.tags[] | startswith(\"${PREFIX}\")] | all)
        | [.id, .created_at, (.metadata.container.tags | join(\",\"))]
        | @tsv" 2>&1)"
if [ $? -ne 0 ]; then
    log "listing package versions failed; skipping"
    log "${versions}"
    exit 0
fi
if [ -z "$versions" ]; then
    log "no ${PREFIX}* versions found; nothing to do"
    exit 0
fi

sorted="$(printf '%s\n' "$versions" | sort -k2,2r)"
log "found $(printf '%s\n' "$sorted" | wc -l | tr -d ' ') ${PREFIX}* version(s)"

deleted=0
kept=0
while IFS=$'\t' read -r id created tags; do
    [ -n "$id" ] || continue
    if [ "$kept" -lt "$KEEP" ] || [[ ",${tags}," == *",${current},"* ]]; then
        kept=$((kept + 1))
        log "keep   ${tags} (${created})"
        continue
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        log "would delete ${tags} (${created}, version ${id})"
        deleted=$((deleted + 1))
        continue
    fi
    if gh api --method DELETE \
        "/orgs/${OWNER}/packages/container/${PACKAGE}/versions/${id}" > /dev/null 2>&1; then
        log "delete ${tags} (${created})"
        deleted=$((deleted + 1))
    else
        # Most likely the token lacks package-delete rights. Say so once per
        # version and carry on; a full image is still cheaper than a red build.
        log "WARNING: could not delete ${tags} (version ${id})"
    fi
done <<< "$sorted"

log "kept ${kept}, deleted ${deleted}"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    echo "**Pruned TokenSpeed images:** kept ${kept}, deleted ${deleted}" >> "$GITHUB_STEP_SUMMARY"
fi
