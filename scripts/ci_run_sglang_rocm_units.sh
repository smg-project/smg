#!/bin/bash
# Retain the complete installed-engine unit selection and bounded diagnostics.
set -euo pipefail
mkdir -p unit-junit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
    python3 -c 'import os, uuid; output = open(os.environ["GITHUB_OUTPUT"], "a"); output.write("artifact=sglang-unit-results-" + uuid.uuid4().hex + "\n"); output.close()'
fi
pip_status=0
python3 -m pip check >unit-junit/pip.stdout 2>unit-junit/pip.stderr || pip_status=$?
cohort_status=0
bash "$SCRIPT_DIR/ci_setup_sglang_rocm_image.sh" --verify-only unit-junit/native-environment.json \
    >unit-junit/cohort.stdout 2>unit-junit/cohort.stderr || cohort_status=$?
provenance_status=0
python3 "$SCRIPT_DIR/ci_verify_sglang_rocm_e2e.py" --verify "${SMG_CI_SGLANG_E2E_SNAPSHOT:-}" --root "$SCRIPT_DIR/.." \
    >unit-junit/provenance.json 2>unit-junit/provenance.stderr || provenance_status=$?
printf '{"pip_returncode":%s,"cohort_returncode":%s,"provenance_returncode":%s}\n' \
    "$pip_status" "$cohort_status" "$provenance_status" >unit-junit/environment-status.json
if [ "$pip_status" -ne 0 ]; then exit "$pip_status"; fi
if [ "$cohort_status" -ne 0 ]; then exit "$cohort_status"; fi
if [ "$provenance_status" -ne 0 ]; then exit "$provenance_status"; fi
timeout --signal=INT --kill-after=30s 600s \
    pytest -vv grpc_servicer/tests -p no:cacheprovider -rs \
        -o faulthandler_timeout=120 --junitxml=unit-junit/sglang.xml
test -s unit-junit/sglang.xml
