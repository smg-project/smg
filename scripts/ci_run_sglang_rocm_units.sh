#!/bin/bash
# Retain the complete installed-engine unit selection and bounded diagnostics.
set -euo pipefail
mkdir -p unit-junit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
    python3 -c 'import os, uuid; output = open(os.environ["GITHUB_OUTPUT"], "a"); output.write("artifact=sglang-unit-results-" + uuid.uuid4().hex + "\n"); output.close()'
fi
bash "$SCRIPT_DIR/ci_setup_sglang_rocm_image.sh" --verify-only unit-junit/native-environment.json
timeout --signal=INT --kill-after=30s 600s \
    pytest -vv grpc_servicer/tests -p no:cacheprovider -rs \
        -o faulthandler_timeout=120 --junitxml=unit-junit/sglang.xml
test -s unit-junit/sglang.xml
