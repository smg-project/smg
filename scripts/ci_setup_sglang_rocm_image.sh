#!/bin/bash
# Set up a job that runs inside SGLang's ROCm image (the AMD benchmark row).
#
# The image provides SGLang, ROCm Torch and Mooncake in /opt/venv. Keep that
# stack, align its native FFI dependencies and install SMG's gRPC packages
# in the same interpreter. A venv of the job's own would not see
# SGLang: --system-site-packages exposes the base interpreter's packages, not
# /opt/venv's. pip keeps to the PIP_CONSTRAINT with which the image pins its
# ROCm torch.

set -euo pipefail

verify_only=0
verify_output=()
if [ "$#" -gt 0 ]; then
    if [ "$1" != --verify-only ] || [ "$#" -gt 2 ]; then
        echo "usage: $0 [--verify-only [REPORT.json]]" >&2
        exit 2
    fi
    verify_only=1
    [ "$#" -eq 1 ] || verify_output=("$2")
    case "${BASH_SOURCE[0]}" in
        */*) pin_helper_directory="${BASH_SOURCE[0]%/*}" ;;
        *) pin_helper_directory=. ;;
    esac
    source "$pin_helper_directory/ci_setup_sglang_rocm_image_pin.sh"
    image_pin="$(read_sglang_rocm_image_pin "$pin_helper_directory/ci_verify_sglang_rocm_cohort.py")"
    if [ "${SMG_SGLANG_ROCM_IMAGE:-}" != "$image_pin" ]; then
        echo "unsupported E2E SGLang image: observed ${SMG_SGLANG_ROCM_IMAGE-}; expected $image_pin" >&2
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY="bash ${SCRIPT_DIR}/ci_retry.sh"

# Preserve this image's companions instead of the generic protobuf-6 fallback.
SMG_CI_GRPC_COMPANION_VERSION="$(python3 -c '
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

try:
    print(Version(version("grpcio-reflection")))
except PackageNotFoundError:
    raise SystemExit("ERROR: SGLang ROCm image lacks grpcio-reflection metadata")
')"

echo "image: $(python3 -V) at $(command -v python3)"

# This repair is scoped to the pinned SGLang 0.5.21 / ROCm 10 image cohort.
# Build the original extension sdist against this image's Torch builtin API;
# a cached or public binary wheel may have been built for a different Torch.
python3 -c '
from importlib.metadata import version
import torch

sglang_version = version("sglang")
torch_version = version("torch")
hip_version = torch.version.hip
if (sglang_version != "0.5.21"
        or torch_version != "2.11.0+rocm10.0.0" or not hip_version):
    raise SystemExit(
        "ERROR: unsupported SGLang ROCm native dependency cohort: "
        "observed sglang=%s torch=%s hip=%r; "
        "expected sglang=0.5.21 torch=2.11.0+rocm10.0.0 and HIP enabled. "
        "Update TVM/TileLang/DLPack pins in "
        "scripts/ci_setup_sglang_rocm_image.sh with the image."
        % (sglang_version, torch_version, hip_version)
    )
api = getattr(torch.Tensor, "__dlpack_c_exchange_api__", None)
if api is None or type(api).__name__ != "PyCapsule":
    raise SystemExit(
        "ERROR: SGLang ROCm image lacks the Torch builtin DLPack C API: "
        "expected PyCapsule, observed %s" % type(api).__name__
    )
'
if [ "$verify_only" -eq 0 ]; then
    $RETRY 3 10 python3 -m pip install --no-cache-dir --no-build-isolation \
        --no-binary=torch-c-dlpack-ext \
        "apache-tvm-ffi==0.1.12" "tilelang==0.1.15" \
        "torch-c-dlpack-ext @ https://files.pythonhosted.org/packages/37/de/921b6491efce5c389a5ef9bbed3d2d6660005840dae488124173180859ab/torch_c_dlpack_ext-0.1.5.tar.gz#sha256=d06f0357d575d22a168cc77acb9020fc4bae30968ceb6718a055dcbe92bacabe"

    # Install gRPC packages from source (not PyPI) so PR changes are always tested
    $RETRY 3 10 python3 -m pip install -e crates/grpc_client/python/ -e grpc_servicer/

fi

# Check selected NIXL in both setup and verify-only mode before publishing
# either a verified cohort report or the next-step companion handoff.
if [ "${E2E_KV_BACKEND:-}" = "nixl" ] || [ "${E2E_SGLANG_TRANSFER_BACKEND:-}" = "nixl" ]; then
    python3 -c "import torch; from nixl._api import nixl_agent; from nixl._bindings import nixlRemoteDisconnectError"
fi

python3 "$SCRIPT_DIR/ci_verify_sglang_rocm_cohort.py" "${verify_output[@]}"

# Carry the observed image version into the next Actions step.
if [ "$verify_only" -eq 0 ] && [ -n "${GITHUB_ENV:-}" ]; then
    echo "SMG_CI_GRPC_COMPANION_VERSION=${SMG_CI_GRPC_COMPANION_VERSION}" >> "$GITHUB_ENV"
fi
