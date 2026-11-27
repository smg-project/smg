#!/bin/bash
# Delete superseded prebuilt GPU backend carrier and combined runner images from GHCR.
#
# Each prefix (`ci-gpu-backend-vllm-`, `ci-gpu-backend-sglang-`,
# `ci-gpu-backend-trtllm-`, `ci-gpu-runner-`) is pruned independently so
# rebuilding one backend never evicts another. For each prefix, the currently
# resolved tag is always protected, plus the newest SMG_GPU_IMAGE_KEEP (default
# 2) versions so in-flight PRs on a recent base commit still find their image.
#
# Only versions whose tags ALL carry that exact prefix are eligible. Release,
# nightly, and TokenSpeed tags in the same GHCR package are never touched.
#
# TOLERANT BY DESIGN: cleanup must never fail the publish it follows.
#
# Env knobs:
#   SMG_GPU_IMAGE_KEEP  how many recent versions to keep per prefix (default 2)
#   DRY_RUN=1           list what would be deleted, delete nothing

set -uo pipefail # deliberately NOT -e: every failure is a soft skip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEEP="${SMG_GPU_IMAGE_KEEP:-2}"
OWNER="${GITHUB_REPOSITORY_OWNER:-smg-project}"
PACKAGE="${SMG_GPU_IMAGE_PACKAGE:-smg}"

DEFAULT_RUNNER_BASE="ghcr.io/actions/actions-runner:2.337.0@sha256:5036480998280bb21e32ade9fe1b02b493861ac314b62ba1aea320b94f56ec97"
DEFAULT_CARRIER_BASE="ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55"
RUNNER_BASE="${SMG_CI_GPU_RUNNER_BASE_IMAGE:-$DEFAULT_RUNNER_BASE}"
CARRIER_BASE="${SMG_CI_GPU_CARRIER_BASE_IMAGE:-$DEFAULT_CARRIER_BASE}"
DRIVER_VER="${SMG_CI_GPU_RUNNER_CUDA_DRIVER_VERSION:-580.65.06}"

log() { echo "[prune-gpu-runner-images] $*"; }

if ! [[ "$KEEP" =~ ^[0-9]+$ ]] || [ "$KEEP" -lt 1 ]; then
    log "SMG_GPU_IMAGE_KEEP=${KEEP} is not a positive integer; skipping"
    exit 0
fi
if ! command -v gh &> /dev/null; then
    log "gh CLI not available; skipping"
    exit 0
fi

# Fetch all package versions once; filter per prefix locally to avoid extra API calls.
all_versions="$(gh api --paginate \
    "/orgs/${OWNER}/packages/container/${PACKAGE}/versions?per_page=100" \
    -q '.[] | select(.metadata.container.tags | length > 0) | [.id, .created_at, (.metadata.container.tags | join(","))] | @tsv' 2>&1)"
if [ $? -ne 0 ]; then
    log "listing package versions failed; skipping"
    log "${all_versions}"
    exit 0
fi

total_kept=0
total_deleted=0

prune_prefix() {
    local prefix="$1" current="$2"
    if [ -z "$current" ]; then
        log "could not resolve current tag for ${prefix}*; skipping prefix"
        return 0
    fi
    log "prefix ${prefix}*: current=${current} (keeping newest ${KEEP})"

    local matches sorted kept=0 deleted=0
    matches="$(awk -F'\t' -v p="$prefix" '
        {
            n = split($3, tags, ",")
            if (n == 0) next
            ok = 1
            for (i = 1; i <= n; i++) {
                if (index(tags[i], p) != 1) { ok = 0; break }
            }
            if (ok) print $0
        }
    ' <<< "$all_versions")"
    if [ -z "$matches" ]; then
        log "no ${prefix}* versions found"
        return 0
    fi

    sorted="$(printf '%s\n' "$matches" | sort -t $'\t' -k2,2r)"
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
        if err="$(gh api --method DELETE \
            "/orgs/${OWNER}/packages/container/${PACKAGE}/versions/${id}" 2>&1 > /dev/null)"; then
            log "delete ${tags} (${created})"
            deleted=$((deleted + 1))
        else
            log "WARNING: could not delete ${tags} (version ${id}): ${err//$'\n'/ }"
        fi
    done <<< "$sorted"

    total_kept=$((total_kept + kept))
    total_deleted=$((total_deleted + deleted))
    log "prefix ${prefix}*: kept ${kept}, deleted ${deleted}"
}

for backend in vllm sglang trtllm; do
    current_tag="$(python3 "${SCRIPT_DIR}/ci_gpu_runner_image.py" tag \
        --backend "$backend" \
        --base-image "$CARRIER_BASE" \
        --cuda-driver-version "$DRIVER_VER" 2> /dev/null || true)"
    prune_prefix "ci-gpu-backend-${backend}-" "$current_tag"
done

runner_tag="$(python3 "${SCRIPT_DIR}/ci_gpu_runner_image.py" tag \
    --backend all \
    --base-image "$RUNNER_BASE" \
    --carrier-base-image "$CARRIER_BASE" \
    --cuda-driver-version "$DRIVER_VER" 2> /dev/null || true)"
prune_prefix "ci-gpu-runner-" "$runner_tag"

log "total: kept ${total_kept}, deleted ${total_deleted}"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    echo "**Pruned GPU runner/carrier images:** kept ${total_kept}, deleted ${total_deleted}" >> "$GITHUB_STEP_SUMMARY"
fi
