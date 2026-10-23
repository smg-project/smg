#!/usr/bin/env bash
set -euo pipefail
python3 .github/amd-vllm-python/cairo_repair.py preflight
source_url='https://files.pythonhosted.org/packages/bb/5e/19ab2572981a996ce96b853a189d2dcced40506b67d9555a7568457abcf2/pycairo-1.29.2.tar.gz#sha256=3e69fff74fe64f5ba2dfa31f67c6bdf26413342574047437d2ac520d35e9a489'
python3 -m pip wheel --retries 0 --no-cache-dir --no-deps --no-binary=pycairo --wheel-dir diagnostic-results/pycairo/wheel "$source_url" > diagnostic-results/pycairo/build.log 2>&1
python3 -m pip install --retries 0 --no-deps diagnostic-results/pycairo/wheel/pycairo-1.29.2-*.whl > diagnostic-results/pycairo/install.log 2>&1
python3 .github/amd-vllm-python/cairo_repair.py verify
