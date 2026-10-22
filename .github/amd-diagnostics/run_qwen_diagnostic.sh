#!/usr/bin/env bash
set -euo pipefail

python3 .github/amd-diagnostics/reuse_foundation.py
python3 -m pip install --retries 0 -e crates/grpc_client/python/ -e grpc_servicer/
python3 -m pip uninstall -y smg
python3 -m pip install --retries 0 wheel/*.whl e2e_test/ clients/python/
python3 -m pip install --retries 0 'grpcio-health-checking==1.81.*' 'grpcio-reflection==1.81.*'
python3 -c "import torch, sglang, mooncake, smg_grpc_servicer; assert torch.version.hip; assert sglang.__version__ == '0.5.21'; print('sglang', sglang.__version__, 'HIP', torch.version.hip)"
python3 -m pip freeze > diagnostic-results/pip-freeze.txt
python3 -m pytest .github/amd-diagnostics/test_qwen_vision_routes.py \
  --confcutdir=.github/amd-diagnostics -o addopts='' -p no:rerunfailures \
  -s -vv --junitxml=diagnostic-results/diagnostic.xml
