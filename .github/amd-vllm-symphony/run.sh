#!/usr/bin/env bash
set -euo pipefail
python3 .github/amd-vllm-symphony/reuse.py
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/image-before.json
python3 .github/amd-vllm-symphony/environment.py before
export PIP_CONSTRAINT="${PIP_CONSTRAINT:+$PIP_CONSTRAINT }$PWD/diagnostic-results/runtime-pins.txt"
test "$E2E_ENGINE" = vllm
source scripts/ci_setup_vllm_rocm_image.sh
export SMG_CI_GRPC_COMPANION_VERSION
bash .github/amd-vllm-symphony/build_cairo.sh
python3 -m pip uninstall -y smg
python3 -m pip install --retries 0 wheel/*.whl
bash scripts/ci_install_e2e_deps.sh sentence-transformers pytest-asyncio
touch diagnostic-results/install-complete
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/installed.json
python3 .github/amd-vllm-symphony/environment.py installed
python3 -m pip check > diagnostic-results/pipcheck-before.txt 2>&1
python3 -c 'import vllm, msgspec, zmq, torch; assert torch.version.hip; print(vllm.__version__, torch.version.hip)'
python3 -m pytest -q grpc_servicer/tests -p no:cacheprovider -rs --junitxml=diagnostic-results/servicer-engine.xml
python3 .github/amd-qa/closure_audit.py require-vllm diagnostic-results/servicer-engine.xml
python3 -m pip freeze > diagnostic-results/pip-freeze.txt
python3 .github/amd-vllm-symphony/preflight.py
# The 15-second grace is inside the AMD 45-minute test cap; the NV reference keeps 24.
timeout --signal=INT --kill-after=15s 2685s python3 .github/amd-vllm-symphony/run_tests.py
