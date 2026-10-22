#!/usr/bin/env bash
set -euo pipefail

python3 .github/amd-cairo-product/proof.py before
status=0
python3 -m pip check > cairo-results/pip-check-before.txt 2>&1 || status=$?
printf '%s\n' "$status" > cairo-results/pip-check-before-status.txt

finish() {
    status=$?
    trap - EXIT
    set +e
    printf '%s\n' "$status" > cairo-results/helper-exit.txt
    python3 .github/amd-cairo-product/proof.py after || status=1
    python3 .github/amd-cairo-product/proof.py native || status=1
    python3 -m pip check > cairo-results/pip-check-after.txt 2>&1 || status=1
    python3 .github/amd-cairo-product/proof.py devices-after || status=1
    exit "$status"
}
trap finish EXIT

source scripts/ci_setup_vllm_rocm_image.sh
