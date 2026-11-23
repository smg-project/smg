#!/usr/bin/env bash
# Keep the official BFCL client's pinned NumPy out of the serving environment.
set -euo pipefail

CLIENT_ENV="${1:?usage: setup_client_env.sh /absolute/new/client-venv}"
: "${GITHUB_ENV:?GITHUB_ENV is required to publish the client executables}"
if [[ "$CLIENT_ENV" != /* || -e "$CLIENT_ENV" || -L "$CLIENT_ENV" ]]; then
  echo "BFCL client environment must be a new absolute path: $CLIENT_ENV" >&2
  exit 2
fi

ENGINE_PYTHON="$(command -v python)"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
[[ -x "$UV_BIN" ]] || { echo "The vLLM setup must provide uv" >&2; exit 2; }
ENGINE_NUMPY_BEFORE="$("$ENGINE_PYTHON" -c 'from importlib.metadata import version; print(version("numpy"))')"

# No system-site-packages: bfcl-eval requires NumPy 1.26.4, while the current
# NVIDIA engine's ml-dtypes/OpenCV cohort requires NumPy >=2. Client-side Torch
# is only a BFCL dependency; it must not download a second GPU runtime.
"$UV_BIN" venv --python "$ENGINE_PYTHON" "$CLIENT_ENV"
"$UV_BIN" pip install --python "$CLIENT_ENV/bin/python" --torch-backend=cpu \
  "bfcl-eval==2026.3.23" soundfile
"$UV_BIN" pip check --python "$CLIENT_ENV/bin/python"
"$CLIENT_ENV/bin/python" -I - <<'PY'
import importlib.metadata as metadata
import sys

import torch

assert sys.prefix != sys.base_prefix
assert metadata.version("bfcl-eval") == "2026.3.23"
assert torch.version.cuda is None and torch.version.hip is None
print("BFCL client:", sys.executable, "numpy", metadata.version("numpy"), "torch CPU")
PY

ENGINE_NUMPY_AFTER="$("$ENGINE_PYTHON" -c 'from importlib.metadata import version; print(version("numpy"))')"
[[ "$ENGINE_NUMPY_AFTER" == "$ENGINE_NUMPY_BEFORE" ]]
# Exercise the import chain that failed before either NVIDIA arm could start.
"$ENGINE_PYTHON" -c 'import numpy; import vllm.entrypoints.cli.launch; print("BFCL engine launch import canary: numpy", numpy.__version__)'
[[ -x "$CLIENT_ENV/bin/bfcl" ]]

# Do not activate the client or add it to PATH: both serving arms, SMG and their
# subprocesses must keep the original engine interpreter and dependencies.
printf 'BFCL_EXECUTABLE=%s/bin/bfcl\nBFCL_CLIENT_PYTHON=%s/bin/python\n' \
  "$CLIENT_ENV" "$CLIENT_ENV" >> "$GITHUB_ENV"
