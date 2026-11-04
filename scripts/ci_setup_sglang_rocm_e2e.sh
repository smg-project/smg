#!/bin/bash
# Official-cohort E2E setup, retaining the original Torch and RPC provenance proof.
set -euo pipefail
case "${BASH_SOURCE[0]}" in
    */*) script_parent="${BASH_SOURCE[0]%/*}" ;;
    *) script_parent=. ;;
esac
source "$script_parent/ci_setup_sglang_rocm_image_pin.sh"
image_pin="$(read_sglang_rocm_image_pin "$script_parent/ci_verify_sglang_rocm_cohort.py")"
if [ "${SMG_SGLANG_ROCM_IMAGE-}" != "$image_pin" ]; then
    echo "unsupported E2E SGLang image: observed ${SMG_SGLANG_ROCM_IMAGE-}; expected $image_pin" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$script_parent" && pwd)"
mkdir -p e2e-logs
snapshot=e2e-logs/sglang-rocm-e2e-baseline.json
python3 "$SCRIPT_DIR/ci_verify_sglang_rocm_e2e.py" --snapshot "$snapshot"
bash "$SCRIPT_DIR/ci_setup_sglang_rocm_image.sh"
python3 "$SCRIPT_DIR/ci_verify_sglang_rocm_e2e.py" --verify "$snapshot" --capture-rpc --root "$SCRIPT_DIR/.."
python3 -m pip check
if [ -n "${GITHUB_ENV:-}" ]; then
    printf 'SMG_CI_SGLANG_E2E_SNAPSHOT=%s\n' "$snapshot" >> "$GITHUB_ENV"
fi
