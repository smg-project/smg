#!/usr/bin/env bash
set -euo pipefail

python3 .github/amd-pd-env/reuse_foundation.py
python3 .github/amd-pd-env/environment_wheel.py
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/image-before.json
python3 .github/amd-pd-env/runtime_environment.py before
export PIP_CONSTRAINT="${PIP_CONSTRAINT:+$PIP_CONSTRAINT }$PWD/diagnostic-results/runtime-pins.txt"

# Same resolver roots as the reviewed ROCm helper, with one attempt per command.
versions="$(python3 -c 'from importlib.metadata import version; from packaging.version import Version; print(Version(version("vllm")), Version(version("grpcio-reflection")))')"
read -r image_vllm image_companions <<< "$versions"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --retries 0 "vllm==$image_vllm" -e crates/grpc_client/python/ -e grpc_servicer/
python3 -m pip install --retries 0 --ignore-installed pytest
python3 -m pip install --retries 0 --no-deps diagnostic-results/pycairo/wheel/pycairo-1.29.2-*.whl
python3 .github/amd-pd-env/native_wheel.py
python3 -m pip uninstall -y smg
python3 -m pip install --retries 0 wheel/*.whl e2e_test/ clients/python/ pytest-asyncio
python3 -m pip install --retries 0 "grpcio-health-checking==$image_companions" "grpcio-reflection==$image_companions"
touch diagnostic-results/install-complete
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/installed-packages.json
python3 .github/amd-pd-env/runtime_environment.py installed
python3 -m pip check > diagnostic-results/pip-check-before.txt 2>&1
cat diagnostic-results/pip-check-before.txt
python3 -c 'import torch, vllm, nixl, smg_grpc_servicer; assert torch.version.hip; print("vllm", vllm.__version__, "HIP", torch.version.hip)'
python3 .github/amd-pd-env/probe_evidence.py preflight
python3 -m pytest e2e_test/router/test_pd_topologies.py -k TestPDDecodeAbortProbe \
  --reruns 0 -o addopts='' -p no:cacheprovider -s -vv \
  --junitxml=diagnostic-results/pd-close.xml 2>&1 | tee diagnostic-results/pd-close.log
python3 .github/amd-pd-env/probe_evidence.py results
