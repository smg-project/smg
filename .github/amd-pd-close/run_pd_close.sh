#!/usr/bin/env bash
set -euo pipefail

python3 .github/amd-pd-close/reuse_foundation.py

# Same resolver roots as the reviewed ROCm helper, with one attempt per command.
versions="$(python3 -c 'from importlib.metadata import version; from packaging.version import Version; print(Version(version("vllm")), Version(version("grpcio-reflection")))')"
read -r image_vllm image_companions <<< "$versions"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --retries 0 "vllm==$image_vllm" -e crates/grpc_client/python/ -e grpc_servicer/
python3 -m pip install --retries 0 --ignore-installed pytest
python3 -m pip install --retries 0 wheel/*.whl e2e_test/ clients/python/ pytest-asyncio
python3 -m pip install --retries 0 "grpcio-health-checking==$image_companions" "grpcio-reflection==$image_companions"
python3 -c 'import torch, vllm, nixl, smg_grpc_servicer; assert torch.version.hip; print("vllm", vllm.__version__, "HIP", torch.version.hip)'
python3 .github/amd-qa/closure_audit.py snapshot diagnostic-results/installed-packages.json
python3 -m pip check > diagnostic-results/pip-check-before.txt 2>&1
cat diagnostic-results/pip-check-before.txt
python3 .github/amd-pd-close/probe_evidence.py preflight
python3 -m pytest e2e_test/router/test_pd_topologies.py -k TestPDDecodeAbortProbe \
  --reruns 0 -o addopts='' -p no:cacheprovider -s -vv \
  --junitxml=diagnostic-results/pd-close.xml 2>&1 | tee diagnostic-results/pd-close.log
python3 .github/amd-pd-close/probe_evidence.py results
python3 -m pip check > diagnostic-results/pip-check-final.txt 2>&1
cat diagnostic-results/pip-check-final.txt
