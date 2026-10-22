#!/usr/bin/env bash
set -euo pipefail
python3 .github/amd-gateway/reuse.py
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/image-before.json
case "$E2E_ENGINE" in
  vllm)
    source scripts/ci_setup_vllm_rocm_image.sh
    export SMG_CI_GRPC_COMPANION_VERSION
    ;;
  sglang) bash scripts/ci_setup_sglang_rocm_image.sh ;;
  *) exit 1 ;;
esac
python3 -m pip uninstall -y smg
python3 -m pip install --retries 0 wheel/*.whl
bash scripts/ci_install_e2e_deps.sh
python3 -m pip check > diagnostic-results/pipcheck-before.txt 2>&1
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/installed.json
touch diagnostic-results/install-complete
if [ "$E2E_ENGINE" = vllm ]; then
  python3 -c 'import vllm, msgspec, zmq, torch; assert torch.version.hip; print(vllm.__version__, torch.version.hip)'
  python3 -m pytest -q grpc_servicer/tests -p no:cacheprovider -rs --junitxml=diagnostic-results/servicer-engine.xml
  python3 .github/amd-qa/closure_audit.py require-vllm diagnostic-results/servicer-engine.xml
else
  python3 -c 'import sglang, torch; assert sglang.__version__ == "0.5.21"; assert torch.version.hip; print(sglang.__version__, torch.version.hip)'
  python3 .github/amd-gateway/collect_sglang_source.py --output diagnostic-results/sglang-source
fi
python3 -m pip freeze > diagnostic-results/pip-freeze.txt
python3 .github/amd-gateway/run_tests.py
