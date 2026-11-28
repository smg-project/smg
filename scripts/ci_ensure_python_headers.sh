#!/bin/bash
# Guarantee the CPython development headers and host C/C++ compiler for the CI
# venv's interpreter.
#
# Triton, torch's cpp_extension, and nvcc (SGLang/FlashInfer/TRT-LLM) compile C
# and C++ sources against the interpreter's headers at RUNTIME, not at install
# time. Without Python.h or gcc/g++, workers die during startup JIT builds with:
#
#   fatal error: Python.h: No such file or directory
#   RuntimeError: Failed to find C compiler. Please specify via CC environment variable
#   gcc: No such file or directory / nvcc fatal: Failed to preprocess host compiler properties
#
# (TokenSpeed reports the missing header as "Triton is not supported on the
# current platform", and go-bindings-e2e silently falls back to CGO_ENABLED=0
# when gcc is absent.)
#
# Both Python.h and gcc/g++ used to arrive by accident on bare pods via the
# Recommends of `apt-get install -y python3-pip` in ci_setup_python_venv.sh's
# repair path (or via apt-installed cuda-nvcc/cuda-toolkit). Adopting a prebuilt
# carrier venv + NVMe-cached CUDA toolkit skips both apt installs, so install
# whichever host pieces are missing explicitly here.
#
# Called by the engine install scripts (ci_install_{vllm,sglang,trtllm,
# tokenspeed}.sh) right after they activate the venv. Deliberately NOT part of
# ci_setup_python_venv.sh: CPU-only lanes (wheel builds) never compile against
# Python.h and must not inherit an apt dependency for it.
#
# Usage: ci_ensure_python_headers.sh [python]
#   python  interpreter to check; defaults to .venv/bin/python when present,
#           else python3. posix_prefix resolves against the BASE interpreter of
#           a venv, which is the include dir Triton hands to gcc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

# Image builds run as root, even when the base image also provides sudo.
if [ "$(id -u)" != 0 ] && command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi

PYTHON="${1:-}"
if [ -z "$PYTHON" ]; then
    if [ -x ".venv/bin/python" ]; then PYTHON=".venv/bin/python"; else PYTHON="python3"; fi
fi

include_dir="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths(scheme="posix_prefix")["include"])')"
missing_pkgs=()
missing_desc=()

if [ ! -f "${include_dir}/Python.h" ]; then
    py_version="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    missing_pkgs+=("python${py_version}-dev")
    missing_desc+=("Python.h (${include_dir})")
fi

if ! command -v gcc &> /dev/null || ! command -v g++ &> /dev/null; then
    missing_pkgs+=(gcc g++ libc6-dev)
    missing_desc+=("gcc/g++")
fi

if [ ${#missing_pkgs[@]} -eq 0 ]; then
    echo "Python headers and C/C++ compiler: present (${include_dir})"
    exit 0
fi

echo "Host JIT toolchain missing (${missing_desc[*]}); installing ${missing_pkgs[*]}"
if ! command -v apt-get &> /dev/null; then
    echo "ERROR: no apt-get to install ${missing_pkgs[*]} with" >&2
    exit 1
fi
export DEBIAN_FRONTEND=noninteractive
bash "${SCRIPT_DIR}/ci_apt_mirror.sh"
$RETRY 3 10 $SUDO apt-get update -qq
$RETRY 3 10 $SUDO apt-get install -y --no-install-recommends "${missing_pkgs[@]}"

# Fail here rather than 20 minutes later inside a Triton/nvcc JIT compile. A miss
# after a successful install means the interpreter is not the distro one (its
# include dir is elsewhere) or the compiler is not on PATH.
if [ ! -f "${include_dir}/Python.h" ]; then
    echo "ERROR: ${include_dir}/Python.h still missing after apt-get install" >&2
    exit 1
fi
if ! command -v gcc &> /dev/null || ! command -v g++ &> /dev/null; then
    echo "ERROR: gcc/g++ still missing from PATH after apt-get install" >&2
    exit 1
fi
echo "Python headers and C/C++ compiler: installed (${include_dir})"
