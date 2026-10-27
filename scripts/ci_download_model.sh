#!/usr/bin/env bash
# Download HuggingFace models with file-locking to prevent concurrent download
# conflicts when multiple CI pods share the same NVMe cache on a node.
#
# Usage:
#   ci_download_model.sh --gpu-tier <tier>                   # Download all models for a GPU tier
#   ci_download_model.sh <model_id> [<model_id> ...]         # Download specific models
#
# Example:
#   ci_download_model.sh --gpu-tier 1
#   ci_download_model.sh Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct

set -euo pipefail

export HF_HOME="${HF_HOME:-/models}"

# Redirect the hf-xet working dir off the shared model cache.
#
# hf-xet (the default huggingface_hub download backend) writes its log files
# and CAS chunk cache under "$HF_HOME/xet" by default. Because HF_HOME points at
# the per-node hostPath model cache that every CI runner pod shares, multiple
# pods racing on that single "$HF_HOME/xet" dir intermittently fail with
# "Permission denied (os error 13)" while writing xet logs / chunks.
#
# Unlike "$HF_HOME/hub" (content-addressed snapshots, guarded by the per-model
# flock below) and "$HF_HOME/.locks", the xet working dir is shared, unlocked,
# and touched by every download regardless of model. Give each job its own xet
# scratch dir off the shared mount; the snapshot cache in "$HF_HOME/hub" stays
# shared, so cross-job model caching is unaffected.
export HF_XET_CACHE="${HF_XET_CACHE:-${RUNNER_TEMP:-/tmp}/hf-xet}"
mkdir -p "$HF_XET_CACHE"

LOCK_DIR="${HF_HOME}/.locks"
MAX_RETRIES=3
RETRY_DELAY=30

resolve_models_for_tier() {
    local tier="$1"
    # A tier-N runner has N GPUs; any model with tp <= N can run on it.
    # PD-disaggregation jobs run on 2-GPU runners with tp=1 models, so an
    # exact ``tp == tier`` match would leave them empty.
    # ROUTER_LOCAL_MODEL_PATH must be unset so model_specs doesn't resolve to local paths.
    ROUTER_LOCAL_MODEL_PATH="" python3 -c "
import sys
from e2e_test.infra.model_specs import MODEL_SPECS
for model_id, spec in MODEL_SPECS.items():
    if spec['tp'] <= int(sys.argv[1]) and not spec.get('skip_tier_download'):
        print(model_id)
" "$tier"
}

download_model() {
    local model_id="$1"
    local model_dir="${HF_HOME}/hub/models--${model_id//\//--}"
    local lock_file="${LOCK_DIR}/${model_id//\//_}.lock"

    echo "Acquiring lock for ${model_id}..."
    (
        # Serialize downloads across pods sharing the same volume
        flock -w 1800 200 || { echo "ERROR: Timed out waiting for lock on ${model_id}"; exit 1; }

        # Log whether model appears cached, but always run hf download to verify
        # integrity (a killed download can leave snapshots/ partially populated).
        local cached=false
        if [ -d "${model_dir}/snapshots" ] && [ -n "$(ls -A "${model_dir}/snapshots/" 2>/dev/null)" ]; then
            echo "Model ${model_id} found in cache, verifying integrity..."
            cached=true
        else
            echo "Model ${model_id} not in cache, downloading..."
        fi
        local attempt=0
        while [ $attempt -lt $MAX_RETRIES ]; do
            attempt=$((attempt + 1))
            if hf download "$model_id" --quiet 2>&1; then
                if [ "$cached" = true ]; then
                    echo "Model ${model_id} already cached, verified OK."
                else
                    echo "Model ${model_id} downloaded successfully."
                fi
                exit 0
            fi
            echo "Download attempt ${attempt}/${MAX_RETRIES} failed for ${model_id}."
            if [ $attempt -lt $MAX_RETRIES ]; then
                # Shared HF locks can belong to another job; age proves neither ownership nor liveness.
                # Preserve shared cache/locks and the original bounded download retry policy.
                echo "Preserving shared HF locks after failed download of ${model_id}; no age-based cleanup."
                echo "Retrying in ${RETRY_DELAY}s..."
                sleep "$RETRY_DELAY"
            fi
        done
        echo "ERROR: Failed to download ${model_id} after ${MAX_RETRIES} attempts."
        exit 1
    ) 200>"$lock_file"
}

# Parse arguments
models=()
if [ "${1:-}" = "--gpu-tier" ]; then
    if [ -z "${2:-}" ]; then
        echo "Usage: $0 --gpu-tier <tier>"
        exit 1
    fi
    resolved_output="$(resolve_models_for_tier "$2")" || {
        echo "ERROR: Failed to resolve models for GPU tier $2"
        exit 1
    }
    while IFS= read -r model; do
        [ -n "$model" ] && models+=("$model")
    done <<<"$resolved_output"
    if [ ${#models[@]} -eq 0 ]; then
        echo "ERROR: No models resolved for GPU tier $2"
        exit 1
    fi
    echo "Resolved ${#models[@]} model(s) for GPU tier $2: ${models[*]}"
elif [ $# -gt 0 ]; then
    models=("$@")
else
    echo "Usage: $0 --gpu-tier <tier> | <model_id> [<model_id> ...]"
    exit 1
fi

mkdir -p "$LOCK_DIR"

for model_id in "${models[@]}"; do
    download_model "$model_id"
done
