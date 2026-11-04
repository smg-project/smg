#!/bin/bash
# Official-cohort E2E setup, retaining the original Torch and RPC provenance proof.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
if [ "${SMG_SGLANG_ROCM_IMAGE-}" != "$IMAGE" ]; then
    echo "unsupported E2E SGLang image: observed ${SMG_SGLANG_ROCM_IMAGE-}; expected $IMAGE" >&2
    exit 1
fi
snapshot="$(mktemp)"
trap 'rm -f "$snapshot"' EXIT
python3 "$SCRIPT_DIR/ci_verify_sglang_rocm_e2e.py" --snapshot "$snapshot"
bash "$SCRIPT_DIR/ci_setup_sglang_rocm_image.sh"
python3 "$SCRIPT_DIR/ci_verify_sglang_rocm_e2e.py" --verify "$snapshot" --root "$SCRIPT_DIR/.."
python3 -m pip check
