#!/usr/bin/env bash
set -euo pipefail
python3 .github/amd-sglang-repair/reuse.py
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/image-before.json
python3 .github/amd-sglang-repair/environment.py before
export PIP_CONSTRAINT="${PIP_CONSTRAINT:+$PIP_CONSTRAINT }$PWD/diagnostic-results/runtime-pins.txt"
test "$E2E_ENGINE" = sglang
export TORCH_EXTENSIONS_DIR="$RUNNER_TEMP/sglang-native-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
touch diagnostic-results/install-started
python3 .github/amd-sglang-repair/wheel_repair.py
source scripts/ci_setup_sglang_rocm_image.sh
python3 -m pip uninstall -y smg
python3 -m pip install --retries 0 wheel/*.whl
bash scripts/ci_install_e2e_deps.sh sentence-transformers
touch diagnostic-results/install-complete
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/installed.json
python3 .github/amd-sglang-repair/environment.py installed
python3 -m pip check > diagnostic-results/pipcheck-before.txt 2>&1
python3 .github/amd-sglang-repair/tvm_slots.py
python3 .github/amd-sglang-repair/run_native.py
python3 -m pytest -q grpc_servicer/tests -p no:cacheprovider -rs --junitxml=diagnostic-results/servicer-engine.xml
python3 -m pip freeze > diagnostic-results/pip-freeze.txt
for suite in completions embeddings router; do
    python3 .github/amd-sglang-repair/run_tests.py "$suite"
done
