#!/usr/bin/env bash
set -euo pipefail
python3 .github/amd-vllm-zmq/reuse.py
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/image-before.json
python3 .github/amd-vllm-zmq/environment.py before
export PIP_CONSTRAINT="${PIP_CONSTRAINT:+$PIP_CONSTRAINT }$PWD/diagnostic-results/runtime-pins.txt"
test "$E2E_ENGINE" = vllm
source scripts/ci_setup_vllm_rocm_image.sh
export SMG_CI_GRPC_COMPANION_VERSION
bash .github/amd-vllm-zmq/build_cairo.sh
python3 -m pip uninstall -y smg
python3 -m pip install --retries 0 wheel/*.whl
bash scripts/ci_install_e2e_deps.sh sentence-transformers pytest-asyncio
touch diagnostic-results/install-complete
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/installed.json
python3 .github/amd-vllm-zmq/environment.py installed
python3 -m pip check > diagnostic-results/pipcheck-before.txt 2>&1
python3 -c 'import vllm, msgspec, zmq, torch; assert torch.version.hip; print(vllm.__version__, torch.version.hip)'
python3 -m pytest -q grpc_servicer/tests -p no:cacheprovider -rs --junitxml=diagnostic-results/servicer-engine.xml
python3 .github/amd-qa/closure_audit.py require-vllm diagnostic-results/servicer-engine.xml
python3 -m pip freeze > diagnostic-results/pip-freeze.txt
python3 .github/amd-vllm-zmq/preflight.py
# The 15-second kill grace is inside the original 40-minute test limit.
timeout --signal=INT --kill-after=15s 2385s python3 .github/amd-vllm-zmq/run_tests.py
