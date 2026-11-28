#!/bin/bash
# Configure persistent node-NVMe JIT compilation caches for GPU CI jobs.
#
# Ephemeral k8s runner pods discard ~/.cache and ~/.triton at job end, forcing
# every GPU lane to re-compile Triton kernels, FlashInfer JIT ops, vLLM
# torch_compile_cache artifacts, TorchInductor graphs, TVM-FFI kernels, and
# CUDA PTX on cold startup.
#
# This script points those caches at the node-shared NVMe mount
# (/models/.ci-cache/jit by default), partitioned by:
#   <engine>-<trust_tier>-<gpu_arch>-<version_hash>
# where <trust_tier> is "trusted" for main/release/tag non-PR events and
# "untrusted" for PR/feature-branch runs, <gpu_arch> encodes the compute
# capability + driver major (NVIDIA) or GCN target + ROCm release (AMD), and
# <version_hash> hashes the exact Python, engine, PyTorch, Triton, FlashInfer,
# TVM-FFI, and nvcc versions. Different engines, tiers, versions, drivers, or
# GPU architectures do not accidentally reuse each other's compiled code.
#
# Usage: ci_setup_jit_cache.sh [engine]
#
# Environment knobs:
#   E2E_ENGINE / $1             engine name (vllm, sglang, trtllm, tokenspeed)
#   E2E_GPU_VENDOR              nvidia (default) or amd
#   SMG_JIT_CACHE_ROOT          override cache root (default: <model-root>/.ci-cache/jit)
#   SMG_JIT_CACHE_TRUST_TIER    override trust tier (default: trusted on main/release/tags, else untrusted)
#   SMG_JIT_CACHE_MIN_FREE_GB   minimum free GiB required on the mount (default: 50)
#   SMG_JIT_CACHE_MAX_AGE_DAYS  evict untouched entries older than N days (default: 7)
#   SMG_JIT_CACHE_PYTHON        override python binary for version inspection
#   SMG_DISABLE_JIT_CACHE=1     skip persistent JIT cache setup
#
# TOLERANT BY DESIGN: never fails the job. If the NVMe mount is absent,
# unwritable, low on free space, or the GPU architecture cannot be identified,
# this script logs why and exits 0, leaving each engine's pod-local defaults.

set -uo pipefail # deliberately NOT -e: every failure is a soft skip

log() { echo "[jit-cache] $*"; }
skip() {
    log "$*; using ephemeral pod-local JIT caches"
    exit 0
}

if [ "${SMG_DISABLE_JIT_CACHE:-0}" = "1" ]; then
    skip "disabled via SMG_DISABLE_JIT_CACHE=1"
fi

ENGINE="${1:-${E2E_ENGINE:-unknown}}"
ENGINE="$(printf '%s' "$ENGINE" | tr -c 'A-Za-z0-9._-' '_')"
GPU_VENDOR="${E2E_GPU_VENDOR:-nvidia}"
MODEL_ROOT="${ROUTER_LOCAL_MODEL_PATH:-${HF_HOME:-/models}}"
CACHE_ROOT="${SMG_JIT_CACHE_ROOT:-${MODEL_ROOT}/.ci-cache/jit}"
MIN_FREE_GB="${SMG_JIT_CACHE_MIN_FREE_GB:-50}"
MAX_AGE_DAYS="${SMG_JIT_CACHE_MAX_AGE_DAYS:-7}"

PYTHON="${SMG_JIT_CACHE_PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x ".venv/bin/python" ]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python3"
    fi
fi
command -v "$PYTHON" &> /dev/null || skip "python interpreter '${PYTHON}' not found"

# Ensure the cache root exists and is writable before inspecting disk space.
if ! mkdir -p "$CACHE_ROOT/.locks" 2> /dev/null; then
    skip "cache root ${CACHE_ROOT} is not writable"
fi
probe="$(mktemp "${CACHE_ROOT}/.write-probe.XXXXXX" 2> /dev/null || true)"
if [ -z "$probe" ]; then
    skip "cache root ${CACHE_ROOT} is not writable"
fi
rm -f "$probe"

# Non-blocking LRU sweep of stale entries untouched for > MAX_AGE_DAYS.
sweep_expired_entries() {
    local keep_name="${1:-}"
    if ! [[ "$MAX_AGE_DAYS" =~ ^[0-9]+$ ]] || [ "$MAX_AGE_DAYS" -le 0 ]; then
        return 0
    fi
    (
        flock -n 200 || exit 0
        while IFS= read -r stale; do
            [ -n "$stale" ] || continue
            if [ -n "$keep_name" ] && [ "$(basename "$stale")" = "$keep_name" ]; then
                continue
            fi
            old="$(mktemp -d "${CACHE_ROOT}/.evict.XXXXXX" 2> /dev/null || true)"
            if [ -n "$old" ] && mv -T "$stale" "$old" 2> /dev/null; then
                rm -rf "$old" 2> /dev/null || true
            else
                rm -rf "$old" 2> /dev/null || true
            fi
        done < <(find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d \
            ! -name '.locks' ! -name '.evict.*' \
            -mtime "+${MAX_AGE_DAYS}" 2> /dev/null)
    ) 200> "${CACHE_ROOT}/.locks/sweep.lock" || true
}

free_space_gb() {
    local avail_kb
    avail_kb="$(df -Pk "$CACHE_ROOT" 2> /dev/null | awk 'NR==2 {print $4}')"
    if [[ "$avail_kb" =~ ^[0-9]+$ ]]; then
        echo $((avail_kb / 1024 / 1024))
    fi
}

# Guard against filling /raid (ci-node-health alarms above 85% disk usage).
# Reclaim expired cache entries first so the mount can self-recover when space
# dips below MIN_FREE_GB because of stale entries.
if [[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]] && [ "$MIN_FREE_GB" -gt 0 ]; then
    avail_gb="$(free_space_gb)"
    if [[ "$avail_gb" =~ ^[0-9]+$ ]] && [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
        sweep_expired_entries ""
        avail_gb="$(free_space_gb)"
        if [[ "$avail_gb" =~ ^[0-9]+$ ]] && [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
            skip "only ${avail_gb} GiB free on ${CACHE_ROOT} (threshold ${MIN_FREE_GB} GiB)"
        fi
    fi
fi

# Separate PR/feature-branch JIT cache keys from main/release keys so jobs on
# different branches do not accidentally reuse each other's compiled entries.
# Note: this is key separation, not an OS permission boundary (pods on a shared
# runner pool mount /models read-write as the same uid).
resolve_trust_tier() {
    if [ -n "${SMG_JIT_CACHE_TRUST_TIER:-}" ]; then
        printf '%s' "$SMG_JIT_CACHE_TRUST_TIER"
        return 0
    fi
    case "${GITHUB_EVENT_NAME:-}" in
        push|schedule|workflow_dispatch|release)
            case "${GITHUB_REF:-}" in
                refs/heads/main|refs/heads/release/*|refs/tags/*)
                    printf 'trusted'
                    return 0
                    ;;
            esac
            ;;
    esac
    printf 'untrusted'
}

TRUST_TIER="$(resolve_trust_tier | tr -c 'A-Za-z0-9._-' '_')"
[ -n "$TRUST_TIER" ] || TRUST_TIER="untrusted"

detect_gpu_arch() {
    if [ "$GPU_VENDOR" = "amd" ]; then
        local arch="" rocm_ver=""
        if command -v rocm_agent_enumerator &> /dev/null; then
            arch="$(rocm_agent_enumerator 2> /dev/null | grep -E '^gfx[0-9a-z]+' | grep -v '^gfx000$' | sort -u | paste -sd '-' || true)"
        fi
        if [ -z "$arch" ] && command -v rocminfo &> /dev/null; then
            arch="$(rocminfo 2> /dev/null | awk '/Name:[[:space:]]+gfx[0-9a-z]+/ && $2 != "gfx000" {print $2}' | sort -u | paste -sd '-' || true)"
        fi
        if [ -z "$arch" ]; then
            arch="$("$PYTHON" -c '
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(getattr(props, "gcnArchName", "").split(":")[0])
' 2> /dev/null || true)"
        fi
        [ -n "$arch" ] || return 1
        if [ -r /opt/rocm/.info/version ]; then
            rocm_ver="$(awk -F'[.-]' '{print $1"."$2}' /opt/rocm/.info/version 2> /dev/null || true)"
        fi
        if [ -n "$rocm_ver" ]; then
            printf '%s-rocm%s' "$arch" "$rocm_ver"
        else
            printf '%s' "$arch"
        fi
        return 0
    fi

    # NVIDIA: query compute capability + driver major via nvidia-smi (no CUDA context).
    if command -v nvidia-smi &> /dev/null; then
        local smi_out arch drv
        smi_out="$(nvidia-smi --query-gpu=compute_cap,driver_version --format=csv,noheader 2> /dev/null || true)"
        if [ -n "$smi_out" ]; then
            arch="$(printf '%s\n' "$smi_out" | awk -F',' '{gsub(/[[:space:].]/, "", $1); if ($1 != "") print "sm"$1}' | sort -u | paste -sd '-' || true)"
            drv="$(printf '%s\n' "$smi_out" | awk -F',' 'NR==1 {gsub(/[[:space:]]/, "", $2); split($2, a, "."); if (a[1] != "") print "drv"a[1]}' || true)"
            if [ -n "$arch" ]; then
                printf '%s%s' "$arch" "${drv:+-$drv}"
                return 0
            fi
        fi
    fi

    # Fallback when nvidia-smi is unavailable.
    local py_arch
    py_arch="$("$PYTHON" -c '
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"sm{major}{minor}")
' 2> /dev/null || true)"
    [ -n "$py_arch" ] || return 1
    printf '%s' "$py_arch"
}

GPU_ARCH="$(detect_gpu_arch || true)"
GPU_ARCH="$(printf '%s' "$GPU_ARCH" | tr -c 'A-Za-z0-9._-' '_')"
[ -n "$GPU_ARCH" ] || skip "could not determine GPU architecture for ${GPU_VENDOR}"

# Collect nvcc version if a CUDA toolkit is present (affects FlashInfer/TVM-FFI JIT).
NVCC_VERSION="none"
NVCC_BIN=""
if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    NVCC_BIN="${CUDA_HOME}/bin/nvcc"
elif [ -x "/usr/local/cuda-13.0/bin/nvcc" ]; then
    NVCC_BIN="/usr/local/cuda-13.0/bin/nvcc"
elif command -v nvcc &> /dev/null; then
    NVCC_BIN="$(command -v nvcc)"
fi
if [ -n "$NVCC_BIN" ]; then
    NVCC_VERSION="$("$NVCC_BIN" --version 2> /dev/null | grep -oE 'release [0-9]+\.[0-9]+(, V[0-9.]+)?' | head -1 || echo unknown)"
fi

MANIFEST="$("$PYTHON" - "$ENGINE" "$TRUST_TIER" "$GPU_VENDOR" "$GPU_ARCH" "$NVCC_VERSION" <<'PY' 2> /dev/null
import importlib.metadata as metadata
import sys

engine, trust_tier, vendor, arch, nvcc = sys.argv[1:6]
lines = [
    f"engine={engine}",
    f"trust_tier={trust_tier}",
    f"gpu_vendor={vendor}",
    f"gpu_arch={arch}",
    f"python={sys.version_info.major}.{sys.version_info.minor}",
    f"nvcc={nvcc}",
]
packages = [
    "vllm",
    "sglang",
    "sgl-kernel",
    "tensorrt-llm",
    "tokenspeed",
    "tokenspeed-engine",
    "tokenspeed-kernel",
    "tokenspeed-triton",
    "torch",
    "triton",
    "flashinfer-python",
    "flashinfer-jit-cache",
    "tvm-ffi",
    "apache-tvm-ffi",
]
for pkg in packages:
    try:
        lines.append(f"pkg.{pkg}={metadata.version(pkg)}")
    except metadata.PackageNotFoundError:
        pass
try:
    import torch
    lines.append(f"torch.cuda={getattr(torch.version, 'cuda', None)}")
    lines.append(f"torch.hip={getattr(torch.version, 'hip', None)}")
except Exception:
    pass
print("\n".join(lines))
PY
)" || skip "failed to inspect installed engine/compiler versions"
[ -n "$MANIFEST" ] || skip "empty version manifest"

VERSION_HASH="$(printf '%s\n' "$MANIFEST" | sha256sum | cut -c 1-16)"
KEY="${ENGINE}-${TRUST_TIER}-${GPU_ARCH}-${VERSION_HASH}"
ENTRY="${CACHE_ROOT}/${KEY}"

subdirs=(triton flashinfer vllm torchinductor tvm-ffi)
if [ "$GPU_VENDOR" != "amd" ]; then
    subdirs+=(cuda)
fi
for d in "${subdirs[@]}"; do
    mkdir -p "${ENTRY}/${d}" 2> /dev/null || skip "could not create ${ENTRY}/${d}"
done

if [ ! -f "${ENTRY}/manifest.txt" ]; then
    tmp_manifest="$(mktemp "${ENTRY}/.manifest.XXXXXX" 2> /dev/null || true)"
    if [ -n "$tmp_manifest" ]; then
        printf '%s\n' "$MANIFEST" > "$tmp_manifest" && mv -f "$tmp_manifest" "${ENTRY}/manifest.txt" || rm -f "$tmp_manifest"
    fi
fi
touch "$ENTRY" 2> /dev/null || true

# Non-blocking LRU sweep of stale entries untouched for > MAX_AGE_DAYS.
sweep_expired_entries "$KEY"

export_var() {
    local name="$1" value="$2"
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "${name}=${value}" >> "$GITHUB_ENV"
    fi
}

export_var "TRITON_CACHE_DIR" "${ENTRY}/triton"
export_var "FLASHINFER_WORKSPACE_BASE" "${ENTRY}/flashinfer"
export_var "VLLM_CACHE_ROOT" "${ENTRY}/vllm"
export_var "TORCHINDUCTOR_CACHE_DIR" "${ENTRY}/torchinductor"
export_var "TORCHINDUCTOR_FX_GRAPH_CACHE" "1"
export_var "TVM_FFI_CACHE_DIR" "${ENTRY}/tvm-ffi"
if [ "$GPU_VENDOR" != "amd" ]; then
    export_var "CUDA_CACHE_PATH" "${ENTRY}/cuda"
    export_var "CUDA_CACHE_MAXSIZE" "4294967296"
fi

log "Enabled persistent JIT cache at ${ENTRY} (trust=${TRUST_TIER}, vendor=${GPU_VENDOR}, arch=${GPU_ARCH})"
