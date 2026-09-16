#!/bin/bash
# Guarantee the CPython development headers for the CI venv's interpreter.
#
# Triton (and torch's cpp_extension) compile C sources against the interpreter's
# headers at RUNTIME, not at install time: the first Triton kernel build looks
# for Python.h and, when it is missing, every vLLM worker dies inside the engine
# profile run with
#
#   fatal error: Python.h: No such file or directory
#   torch._inductor.exc.InductorError: CalledProcessError: gcc ... cuda_utils.c
#
# (TokenSpeed reports the same failure as "Triton is not supported on the
# current platform".)
#
# Whether the headers exist depends on how ci_setup_python_venv.sh obtained the
# interpreter: a uv-provisioned CPython bundles them in its own prefix, while a
# host interpreter only has them if python3.X-dev is installed. That package
# used to arrive by accident, as an apt Recommends of python3-pip pulled in by
# the venv script's repair path, so a host whose `python3 -m venv` just works
# (Ubuntu 24.04 bare-metal runners) never gets it. Install it explicitly.
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

# sudo is absent when this runs as root inside `docker build`; degrade to
# running the commands directly.
if command -v sudo &> /dev/null; then SUDO="sudo"; else SUDO=""; fi

PYTHON="${1:-}"
if [ -z "$PYTHON" ]; then
    if [ -x ".venv/bin/python" ]; then PYTHON=".venv/bin/python"; else PYTHON="python3"; fi
fi

include_dir="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths(scheme="posix_prefix")["include"])')"
if [ -f "${include_dir}/Python.h" ]; then
    echo "Python headers: present at ${include_dir}"
    exit 0
fi

py_version="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Python.h missing from ${include_dir}; installing python${py_version}-dev"
if ! command -v apt-get &> /dev/null; then
    echo "ERROR: no apt-get to install python${py_version}-dev with" >&2
    exit 1
fi
export DEBIAN_FRONTEND=noninteractive
bash "${SCRIPT_DIR}/ci_apt_mirror.sh"
$RETRY 3 10 $SUDO apt-get update -qq
$RETRY 3 10 $SUDO apt-get install -y --no-install-recommends "python${py_version}-dev"

# Fail here rather than 20 minutes later inside a Triton JIT compile. A miss
# after a successful install means the interpreter is not the distro one (its
# include dir is elsewhere), which apt cannot fix.
if [ ! -f "${include_dir}/Python.h" ]; then
    echo "ERROR: python${py_version}-dev did not provide ${include_dir}/Python.h" >&2
    exit 1
fi
echo "Python headers: installed at ${include_dir}"
